#!/usr/bin/env python3
"""键盘 teleop 抓杯仿真（macOS 请用 mjpython 启动）。"""

from __future__ import annotations

from _bootstrap import bootstrap

bootstrap()

from pickcup import main

if __name__ == "__main__":
    main()
