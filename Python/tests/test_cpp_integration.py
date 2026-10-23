"""End-to-end recognition and UART protocol checks using a pseudo-terminal."""
import json
import os
from pathlib import Path
import pty
import select
import subprocess
import tempfile
import unittest

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
EXE = PROJECT / "build" / "ps"


@unittest.skipUnless(EXE.exists(), "build the C++ ps executable first")
class CppIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.master, self.slave = pty.openpty()
        self.serial = os.ttyname(self.slave)

    def tearDown(self):
        os.close(self.master)
        os.close(self.slave)
        self.temp.cleanup()

    def run_cpp(self, *arguments):
        return subprocess.run([str(EXE), *map(str, arguments), "--serial-device", self.serial],
                              capture_output=True, text=True, timeout=45)

    def received(self):
        data = b""
        while select.select([self.master], [], [], 0.05)[0]:
            data += os.read(self.master, 65536)
        return data

    def check_recognition(self, task, image):
        source = self.directory / "scene.png"
        output = self.directory / "output"
        cv2.imwrite(str(source), image)
        result = self.run_cpp(f"--{task}", "--image", source, "--output-dir", output)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.received(), b"", "recognition wrote UART bytes")
        self.assertIsNotNone(cv2.imread(str(output / "a4_warped.jpg")))
        self.assertIsNotNone(cv2.imread(str(output / ("task2_machine.png" if task == "task2" else "lines_machine.png"))))
        packet = json.loads((output / "machine_targets.json").read_text())
        self.assertEqual(packet["task"], task)
        result = self.run_cpp("--send-result", output / "machine_targets.json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        expected = ""
        for target in packet["targets"]:
            ox, oy = target["origin_pixels"]
            tx, ty = target["target_pixels"]
            angle = target["rotation_degrees"]
            index = target["source_index"]
            expected += f"{index} 0 x:{oy/4:.1f} y:{ox/4:.1f} r:{angle:.1f};\n"
            expected += f"{index} 1 x:{ty/4:.1f} y:{tx/4:.1f};\n"
        self.assertEqual(self.received(), (expected + "finish\n").encode())
        # Reuse the same output directory with a failed capture; old targets must disappear.
        failed = self.run_cpp(f"--{task}", "--image", self.directory / "missing.png", "--output-dir", output)
        self.assertEqual(failed.returncode, 1)
        self.assertFalse((output / "machine_targets.json").exists())
        self.assertEqual(self.received(), b"")

    def test_task1_recognition_then_manual_send(self):
        scene = np.full((1400, 1000, 3), 180, np.uint8)
        cv2.rectangle(scene, (80, 100), (920, 1288), (20, 20, 20), -1)
        cv2.rectangle(scene, (180, 280), (340, 560), (245, 245, 245), -1)
        cv2.rectangle(scene, (500, 700), (660, 980), (245, 245, 245), -1)
        self.check_recognition("task1", scene)

    def test_task2_recognition_then_manual_send(self):
        # The existing fixture is already warped. Add a desk border for A4 detection.
        warped = cv2.imread(str(PROJECT / "tests/data/card_three_uneven_light.jpeg"))
        self.assertIsNotNone(warped)
        scene = cv2.copyMakeBorder(warped, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(180, 180, 180))
        self.check_recognition("task2", scene)

    def test_invalid_second_target_sends_no_partial_batch(self):
        target = {"source_index": 0, "origin_pixels": [40, 80], "target_pixels": [120, 160], "rotation_degrees": 90}
        bad_target = dict(target, source_index=1, target_pixels=[-1, 20])
        path = self.directory / "invalid.json"
        path.write_text(json.dumps({"version": 1, "ready": 1, "task": "task1", "targets": [target, bad_target]}))
        result = self.run_cpp("--send-result", path)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.received(), b"")


if __name__ == "__main__":
    unittest.main()
