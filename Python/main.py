#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from src.controller import PuzzleController
from src.ui import UI


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="拼图 UI：后台调用原 C++ 流程，显示 A4 和机械结果")
    parser.add_argument("--exe", type=Path, default=directory.parent / "build" / "ps")
    parser.add_argument("--image", type=Path, help="离线调试图片；不填写则由 C++ 每次重新拍照")
    parser.add_argument("--serial-device", default="/dev/ttyAMA0")
    args = parser.parse_args()
    os.environ.setdefault("DISPLAY", ":0")
    controller = PuzzleController(args.exe, directory / "runs", image=args.image,
                                  serial_device=args.serial_device)
    try:
        UI(controller).run()
    finally:
        controller.close()


if __name__ == "__main__":
    main()
