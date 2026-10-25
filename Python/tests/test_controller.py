import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

from src.controller import PuzzleController


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.exe = self.directory / "fake_cpp"
        self.mode = self.directory / "mode"
        self.mode.write_text("success")
        self.exe.write_text(f"#!{sys.executable}\n" + r'''
import base64, json, os, sys, time
from pathlib import Path
base = Path(__file__).parent
args = sys.argv[1:]
with (base / 'calls').open('a') as log:
    log.write(json.dumps(args) + '\n')
mode = (base / 'mode').read_text()
if '--send-result' in args:
    sys.exit(2 if mode == 'send_error' else 0)
if mode == 'slow':
    time.sleep(0.35)
if mode == 'hang':
    (base / 'pid').write_text(str(os.getpid()))
    time.sleep(60)
if mode == 'fatal':
    sys.exit(2)
counter = base / 'counter'
n = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(n))
if mode == 'retry' and n == 1:
    print('recognition failed')
    sys.exit(1)
output = Path(args[args.index('--output-dir') + 1])
task = 'task2' if '--task2' in args else 'task1'
png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=')
(output / 'a4_warped.jpg').write_bytes(png)
(output / ('task2_machine.png' if task == 'task2' else 'lines_machine.png')).write_bytes(png)
packet = {'version':1, 'ready':1, 'task':task, 'targets':[{'source_index':0}]}
if mode == 'wrong_task':
    packet['task'] = 'other'
(output / 'machine_targets.json').write_text(json.dumps(packet))
''')
        self.exe.chmod(0o700)
        self.controller = PuzzleController(self.exe, self.directory / "runs", retry_delay=0.05,
                                           process_timeout=2)

    def tearDown(self):
        self.controller.close()
        self.temp.cleanup()

    def wait_done(self, timeout=5):
        deadline = time.monotonic() + timeout
        while self.controller.busy and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(self.controller.busy, "background worker did not finish")
        return list(self.controller.events.queue)

    def calls(self):
        return [json.loads(line) for line in (self.directory / "calls").read_text().splitlines()]

    def test_retry_then_only_explicit_send_uses_successful_batch(self):
        self.mode.write_text("retry")
        self.assertTrue(self.controller.start("task2"))
        events = self.wait_done()
        self.assertEqual(len(self.calls()), 2)
        self.assertIn("未成功", " ".join(str(value) for kind, value in events))
        result = self.controller.result
        self.assertEqual(result.machine.name, "task2_machine.png")
        self.assertEqual(result.directory.name, "attempt_0002")
        self.assertNotEqual(self.calls()[0][-1], self.calls()[1][-1])
        self.assertTrue(all("--image" not in call for call in self.calls()))
        self.assertTrue(self.controller.send())
        events = self.wait_done()
        self.assertEqual(self.calls()[-1][0:2], ["--send-result", str(result.targets)])
        self.assertIn(("sent", None), events)
        self.assertFalse(self.controller.send())

    def test_stop_terminates_capture_without_blocking_caller(self):
        self.mode.write_text("hang")
        start = time.monotonic()
        self.assertTrue(self.controller.start("task1"))
        self.assertLess(time.monotonic() - start, 0.2)
        deadline = time.monotonic() + 3
        while not (self.directory / "pid").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        pid = int((self.directory / "pid").read_text())
        self.controller.stop()
        self.assertIn(("stopped", None), self.wait_done())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        self.assertIsNone(self.controller.result)
        self.assertFalse(self.controller.send())
        self.assertEqual(len(self.calls()), 1)

    def test_fatal_error_does_not_loop(self):
        self.mode.write_text("fatal")
        self.controller.start("task1")
        events = self.wait_done()
        self.assertTrue(any(kind == "error" for kind, _ in events))
        self.assertEqual(len(self.calls()), 1)

    def test_new_task_clears_previous_targets_immediately(self):
        self.controller.start("task1")
        self.wait_done()
        self.assertIsNotNone(self.controller.result)
        self.mode.write_text("hang")
        self.assertTrue(self.controller.start("task2"))
        self.assertIsNone(self.controller.result)
        self.assertFalse(self.controller.send())
        self.assertFalse(self.controller.start("task1"))
        self.controller.stop()
        self.wait_done()

    def test_wrong_task_packet_cannot_enable_send(self):
        self.mode.write_text("wrong_task")
        self.controller.start("task1")
        self.assertTrue(any(kind == "error" for kind, _ in self.wait_done()))
        self.assertFalse(self.controller.send())

    def test_failed_send_cannot_be_repeated(self):
        self.controller.start("task1")
        self.wait_done()
        self.mode.write_text("send_error")
        self.assertTrue(self.controller.send())
        self.assertTrue(any(kind == "error" for kind, _ in self.wait_done()))
        self.assertFalse(self.controller.send())
        self.assertEqual(len(self.calls()), 2)

    def test_task2_uses_longer_budget_and_keeps_stop_responsive(self):
        self.mode.write_text("slow")
        self.controller.process_timeout = 0.1
        self.controller.task2_process_timeout = 0.8
        self.assertTrue(self.controller.start("task2"))
        self.assertTrue(any(kind == "ready" for kind, _ in self.wait_done()))
        self.assertEqual(len(self.calls()), 1, "task2 was terminated by the shorter task1 budget")
        self.mode.write_text("hang")
        self.controller.task2_process_timeout = 90
        self.assertTrue(self.controller.start("task2"))
        deadline = time.monotonic() + 3
        while not (self.directory / "pid").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        pid = int((self.directory / "pid").read_text())
        start = time.monotonic()
        self.controller.stop()
        self.assertIn(("stopped", None), self.wait_done())
        self.assertLess(time.monotonic() - start, 1)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_timeout_retries_then_can_be_stopped(self):
        self.mode.write_text("hang")
        self.controller.process_timeout = 0.1
        self.controller.start("task1")
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            if (self.directory / "calls").exists() and len(self.calls()) >= 2:
                break
            time.sleep(0.01)
        self.assertGreaterEqual(len(self.calls()), 2)
        self.controller.stop()
        self.assertIn(("stopped", None), self.wait_done())


if __name__ == "__main__":
    unittest.main()
