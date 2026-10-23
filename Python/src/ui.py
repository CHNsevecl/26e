"""Two result images from the existing C++ recognition pipeline."""
import base64
from pathlib import Path
import queue
import tkinter as tk
from tkinter import ttk

import cv2


class ImagePanel(ttk.Frame):
    def __init__(self, parent, title):
        super().__init__(parent)
        self.title = ttk.Label(self, text=title, anchor="center")
        self.title.pack(fill="x", pady=(0, 4))
        self.canvas = tk.Canvas(self, background="#202020", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.frame = None
        self.photo = None
        self.message = "等待识别结果"
        self.canvas.bind("<Configure>", lambda event: self.render())

    def clear(self, title):
        self.title.configure(text=title)
        self.frame = None
        self.photo = None
        self.render()

    def load(self, path):
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"无法读取图片：{path}")
        self.frame = frame
        self.title.configure(text=Path(path).name)
        self.render()

    def render(self):
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.canvas.delete("all")
        if self.frame is None:
            self.canvas.create_text(width / 2, height / 2, text=self.message, fill="white")
            return
        rows, cols = self.frame.shape[:2]
        ratio = min(width / cols, height / rows)
        image = cv2.resize(self.frame, (max(1, int(cols * ratio)), max(1, int(rows * ratio))),
                           interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
        ok, png = cv2.imencode(".png", image)
        if ok:
            self.photo = tk.PhotoImage(master=self.canvas, data=base64.b64encode(png))
            self.canvas.create_image(width / 2, height / 2, image=self.photo)


class UI:
    def __init__(self, controller, root=None):
        self.controller = controller
        self.root = root if root is not None else tk.Tk()
        self.root.title("拼图识别")
        self.root.geometry("800x600")
        self.root.minsize(640, 480)
        self.closing = False
        self.state = "idle"
        self.displayed_result = None
        self.status = tk.StringVar(value="选择 task1 或 task2 开始识别")
        self.result_path = tk.StringVar(value="")

        content = ttk.Frame(self.root, padding=8)
        content.pack(fill="both", expand=True)
        toolbar = ttk.Frame(content)
        toolbar.pack(fill="x", pady=(0, 8))
        self.task1_button = ttk.Button(toolbar, text="task1", command=lambda: self.start("task1"))
        self.task2_button = ttk.Button(toolbar, text="task2", command=lambda: self.start("task2"))
        self.stop_button = ttk.Button(toolbar, text="停止", command=self.stop, state="disabled")
        self.send_button = ttk.Button(toolbar, text="发送", command=self.send, state="disabled")
        for button in (self.task1_button, self.task2_button, self.stop_button, self.send_button):
            button.pack(side="left", padx=(0, 8), ipadx=12, ipady=5)

        images = ttk.Frame(content)
        images.pack(fill="both", expand=True)
        images.columnconfigure(0, weight=1, uniform="images")
        images.columnconfigure(1, weight=1, uniform="images")
        images.rowconfigure(0, weight=1)
        self.warped_panel = ImagePanel(images, "a4_warped.jpg")
        self.machine_panel = ImagePanel(images, "机械结果")
        self.warped_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.machine_panel.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        ttk.Label(content, textvariable=self.status, wraplength=760).pack(fill="x", pady=(6, 0))
        ttk.Label(content, textvariable=self.result_path, wraplength=760).pack(fill="x")
        self.log = tk.Text(content, height=4, state="disabled", wrap="word")
        self.log.pack(fill="x", pady=(4, 0))
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(50, self.poll)

    def _buttons(self):
        busy = self.state in ("recognizing", "stopping", "sending") or self.controller.busy
        for button in (self.task1_button, self.task2_button):
            button.configure(state="disabled" if busy or self.closing else "normal")
        self.stop_button.configure(state="normal" if self.state == "recognizing" else "disabled")
        self.send_button.configure(state="normal" if self.state == "ready" and not busy and not self.closing else "disabled")

    def start(self, task):
        if self.closing or not self.controller.start(task):
            return
        self.state = "recognizing"
        self.displayed_result = None
        self.warped_panel.clear("a4_warped.jpg")
        self.machine_panel.clear("task2_machine.png" if task == "task2" else "lines_machine.png")
        self.result_path.set("")
        self.status.set(f"{task}：拍照并识别中…")
        self._buttons()

    def stop(self):
        if self.state != "recognizing":
            return
        self.controller.stop()
        self.state = "stopping"
        self.displayed_result = None
        self.status.set("正在停止识别…")
        self._buttons()

    def send(self):
        if self.state != "ready" or not self.controller.send():
            return
        self.state = "sending"
        self.status.set("正在发送本次机械目标…")
        self._buttons()

    def append_log(self, value):
        self.log.configure(state="normal")
        self.log.insert("end", value.rstrip() + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 150:
            self.log.delete("1.0", "80.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def poll(self):
        try:
            while True:
                kind, value = self.controller.events.get_nowait()
                if kind == "log":
                    self.append_log(value)
                elif kind == "status" and self.state == "recognizing":
                    self.status.set(value)
                elif kind == "ready":
                    if self.state == "stopping" or self.closing:
                        self.controller.stop()
                        self.state = "idle"
                        self.status.set("已停止，本次目标已清除")
                        continue
                    try:
                        self.warped_panel.load(value.warped)
                        self.machine_panel.load(value.machine)
                    except (ValueError, cv2.error, tk.TclError) as error:
                        self.controller.stop()
                        self.state = "idle"
                        self.status.set(str(error))
                        continue
                    self.displayed_result = value
                    self.result_path.set(str(value.directory))
                    self.state = "ready"
                    self.status.set(f"{value.task} 识别成功；确认图片后点击“发送”")
                elif kind == "stopped":
                    self.state = "idle"
                    self.status.set("已停止，本次目标已清除")
                elif kind == "error":
                    self.state = "idle"
                    self.status.set(value)
                    self.append_log(value)
                elif kind == "sent":
                    self.state = "sent"
                    self.status.set("本次机械目标已发送")
        except queue.Empty:
            pass
        if self.state == "stopping" and not self.controller.busy:
            self.state = "idle"
            self.status.set("已停止，本次目标已清除")
        if self.closing and not self.controller.busy:
            self.root.destroy()
            return
        self._buttons()
        self.root.after(50, self.poll)

    def close(self):
        self.closing = True
        if self.state == "sending":
            self.status.set("正在发送，发送结束后关闭窗口…")
        else:
            self.controller.stop()
        self._buttons()

    def run(self):
        self.root.mainloop()
