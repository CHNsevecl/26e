# 拼图 UI

源码位于此目录，继续使用 `/home/sevecl/Desktop/PY/Kinter/.zmq_env/bin/python`，原 Kinter 源码保留。

```bash
cmake -S /home/sevecl/Desktop/C/Project/26e -B /home/sevecl/Desktop/C/Project/26e/build
cmake --build /home/sevecl/Desktop/C/Project/26e/build -j2
/home/sevecl/Desktop/C/Project/26e/Python/run.sh
```

- `task1` / `task2` 在后台调用 `../build/ps`，使用现有 C++ 摄像头和识别流程。
- 正常识别失败会等待 1 秒后重新拍照；“停止”结束当前识别并取消重试。
- 成功后显示 `a4_warped.jpg` 和对应的 `lines_machine.png` / `task2_machine.png`。
- 只有手动点击“发送”才发送本次 `machine_targets.json`，识别过程不发送串口。
- 每次尝试保存到独立的 `runs/` 子目录；日志和本次结果目录在 UI 中显示。

可选参数：`--image /现场原图.jpg`（离线调试，重试时重读该文件）、`--serial-device /dev/ttyAMA0`、`--exe /path/to/ps`。

测试：

```bash
cd /home/sevecl/Desktop/C/Project/26e/Python
/home/sevecl/Desktop/PY/Kinter/.zmq_env/bin/python -m unittest discover -s tests -v
```

正常使用时不要同时运行占用同一摄像头的旧 ZMQ 发布程序。此 UI 直接读取 C++ 生成的两张结果图片。
