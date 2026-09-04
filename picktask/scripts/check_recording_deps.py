#!/usr/bin/env python3
"""检查 LeRobot 录制依赖是否满足 pickcup 要求。"""

from __future__ import annotations

import importlib.util
import sys


def _check(name: str, min_version: str | None = None, max_version: str | None = None) -> bool:
    spec = importlib.util.find_spec(name)
    if spec is None:
        print(f"  [缺失] {name}")
        return False
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        print(f"  [错误] {name}: {exc}")
        return False
    version = getattr(mod, "__version__", "?")
    ok = True
    if min_version or max_version:
        from packaging.version import Version

        v = Version(version)
        if min_version and v < Version(min_version):
            print(f"  [过低] {name} {version}，需要 >={min_version}")
            ok = False
        elif max_version and v >= Version(max_version):
            print(f"  [过高] {name} {version}，需要 <{max_version}")
            ok = False
        else:
            print(f"  [OK]   {name} {version}")
    else:
        print(f"  [OK]   {name} {version}")
    return ok


def _check_pytorch_install_source() -> bool:
    """pip 安装的 torch 在 Mac 上常与 conda MKL 冲突。"""
    import subprocess

    def pkg_origin(name: str) -> str | None:
        result = subprocess.run(
            ["conda", "list", name],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in result.stdout.splitlines():
            if line.startswith(name + " "):
                parts = line.split()
                if len(parts) >= 4:
                    return parts[-1]
        return None

    torch_origin = pkg_origin("torch")
    if torch_origin == "pypi":
        print("  [冲突] torch 来自 pip，易缺 libmkl_intel_lp64.2.dylib")
        print("         修复: python -m pip uninstall -y torch torchvision")
        print("               conda install -c pytorch -c conda-forge 'pytorch>=2.7' 'torchvision>=0.22'")
        return False
    if torch_origin:
        print(f"  [OK]   torch 来源: {torch_origin}")
    return True


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})\n")
    print("LeRobot 录制依赖检查：")

    ok = True
    ok &= _check("numpy", "2.0.0", "2.3.0")
    ok &= _check("torch", "2.7.0", None)
    ok &= _check("torchvision", "0.22.0", None)
    ok &= _check_pytorch_install_source()
    ok &= _check("pyarrow")
    ok &= _check("cv2")
    ok &= _check("datasets")
    ok &= _check("av")

    print()
    if ok:
        print("全部通过。可运行: mjpython picktask/scripts/teleop.py")
        return 0

    print("依赖未满足。请在 mujoco_demo 环境中执行：\n")
    print("  pip install -r picktask/requirements-recording.txt\n")
    print("若 pip 报 numpy/torch 冲突，推荐用 conda 一次装齐：\n")
    print('  conda install -c pytorch -c conda-forge \\')
    print('    "numpy>=2.0,<2.3" pytorch torchvision pyarrow opencv av datasets\n')
    print("常见错误：numpy 2.4.x + torch 2.2.x 不兼容 → 需同时升级 torch/torchvision 并降级 numpy 到 2.0–2.2。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
