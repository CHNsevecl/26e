"""Background recognition/retry and explicit sending; no Tk calls in workers."""
from dataclasses import dataclass
import json
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time

import cv2


@dataclass(frozen=True)
class RecognitionResult:
    task: str
    directory: Path
    targets: Path
    warped: Path
    machine: Path


class Cancelled(Exception):
    pass


class ProcessTimeout(Exception):
    pass


class PuzzleController:
    def __init__(self, executable, runs_dir, image=None,
                 serial_device="/dev/ttyAMA0", retry_delay=1.0, process_timeout=45.0,
                 task2_process_timeout=90.0):
        self.executable = Path(executable).resolve()
        self.runs_dir = Path(runs_dir).resolve()
        self.image = Path(image).resolve() if image else None
        self.serial_device = serial_device
        self.retry_delay = retry_delay
        self.process_timeout = process_timeout
        self.task2_process_timeout = task2_process_timeout
        self.events = queue.Queue()
        self.result = None
        self._cancel = threading.Event()
        self._state_lock = threading.Lock()
        self._thread = None
        self._sending = False

    @property
    def busy(self):
        return self._thread is not None and self._thread.is_alive()

    def _post(self, kind, value=None):
        self.events.put((kind, value))

    def start(self, task):
        if self.busy or task not in ("task1", "task2"):
            return False
        self.result = None
        self._cancel = threading.Event()
        self._sending = False
        self._thread = threading.Thread(target=self._recognize, args=(task,), daemon=True)
        self._thread.start()
        return True

    def stop(self):
        # Never cancel an in-progress manual batch: bytes already sent cannot be recalled.
        if not self._sending:
            with self._state_lock:
                self._cancel.set()
                self.result = None

    def _run(self, arguments, directory, log_name="recognition.log", timeout=None):
        if self._cancel.is_set():
            raise Cancelled
        with (directory / log_name).open("w") as log:
            with subprocess.Popen([str(self.executable), *map(str, arguments)],
                                  cwd=directory, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT) as process:
                deadline = time.monotonic() + (self.process_timeout if timeout is None else timeout)
                try:
                    while process.poll() is None:
                        if self._cancel.wait(0.05):
                            raise Cancelled
                        if time.monotonic() >= deadline:
                            raise ProcessTimeout
                    if self._cancel.is_set():
                        raise Cancelled
                    return process.returncode
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=0.8)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()

    @staticmethod
    def _result(task, directory):
        targets = directory / "machine_targets.json"
        with targets.open() as source:
            packet = json.load(source)
        if (packet.get("version") != 1 or packet.get("ready") != 1 or
                packet.get("task") != task or not packet.get("targets")):
            raise ValueError("本次机械目标不完整")
        result = RecognitionResult(task, directory, targets, directory / "a4_warped.jpg",
                                   directory / ("task2_machine.png" if task == "task2" else "lines_machine.png"))
        for path in (result.warped, result.machine):
            if cv2.imread(str(path)) is None:
                raise ValueError(f"结果图片缺失或损坏：{path.name}")
        return result

    def _recognize(self, task):
        try:
            if not self.executable.is_file():
                raise FileNotFoundError(f"请先构建 C++ 程序：{self.executable}")
            if self.image is not None and not self.image.is_file():
                raise FileNotFoundError(f"图片不存在：{self.image}")
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            session = Path(tempfile.mkdtemp(prefix=f"{task}_", dir=self.runs_dir))
            attempt = 0
            while not self._cancel.is_set():
                attempt += 1
                directory = session / f"attempt_{attempt:04d}"
                directory.mkdir()
                self._post("status", f"{task} 第 {attempt} 次：重新拍照并识别中…")
                arguments = [f"--{task}", "--output-dir", directory]
                if self.image is not None:
                    arguments += ["--image", self.image]
                try:
                    timeout = self.task2_process_timeout if task == "task2" else self.process_timeout
                    code = self._run(arguments, directory, timeout=timeout)
                except ProcessTimeout:
                    code = 1
                    self._post("log", "本次识别超时，已结束 C++ 进程。")
                log = directory / "recognition.log"
                if log.exists():
                    self._post("log", log.read_text(errors="replace")[-5000:])
                if code == 0:
                    result = self._result(task, directory)
                    with self._state_lock:
                        if self._cancel.is_set():
                            raise Cancelled
                        self.result = result
                        self._post("ready", result)
                    return
                if code != 1:
                    raise RuntimeError(f"C++ 程序异常退出（{code}），请查看日志：{log}")
                self._post("status", f"{task} 第 {attempt} 次未成功，准备重新拍照；可点击停止。")
                if self._cancel.wait(self.retry_delay):
                    raise Cancelled
            raise Cancelled
        except Cancelled:
            self.result = None
            self._post("stopped")
        except Exception as error:
            self.result = None
            self._post("error", str(error))

    def send(self):
        if self.busy or self.result is None:
            return False
        result = self.result
        self._cancel = threading.Event()
        self._sending = True
        self._thread = threading.Thread(target=self._send, args=(result,), daemon=True)
        self._thread.start()
        return True

    def _send(self, result):
        # Consume once even if sending fails: a partial batch must not be resent blindly.
        self.result = None
        try:
            code = self._run(["--send-result", result.targets, "--serial-device", self.serial_device],
                             result.directory, "send.log")
            self._post("log", (result.directory / "send.log").read_text(errors="replace"))
            if code != 0:
                raise RuntimeError("串口发送失败，请检查设备和发送日志；本次目标已禁用重复发送。")
            self._post("sent")
        except Exception as error:
            self._post("error", f"发送未完成：{error}；请确认机械状态后重新识别。")
        finally:
            self._sending = False

    def close(self):
        self.stop()
        if self._thread:
            self._thread.join(timeout=2)
