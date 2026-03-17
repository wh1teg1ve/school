"""
一键启动电商用户画像系统（Python 版本）

步骤：
1. 安装依赖（requirements.txt）
2. 如有 CSV：可选自动导入到 MySQL（import_csv_to_mysql.py）
3. 启动 Flask（flask_app.py，默认 USE_MYSQL=1）
4. 自动在浏览器打开首页

用法：
    python start_all.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError


def run_step(cmd: list[str], desc: str) -> None:
    """运行一个子进程步骤，仅打印错误，不中断整个流程。"""
    print(f"[执行] {desc} ...")
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"  ⚠ 命令执行返回码 {result.returncode}，可手动检查：{' '.join(cmd)}")
    except Exception as e:
        print(f"  ⚠ 命令执行失败：{e}")


def wait_for_server(url: str, timeout: float = 20.0) -> None:
    """轮询等待 Flask 服务启动，直到可访问或超时。"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urlopen(url) as resp:
                if resp.status == 200:
                    print("Flask 服务已启动。")
                    return
        except URLError:
            pass
        time.sleep(1.0)
    print("  ⚠ 在预期时间内未检测到 Flask 服务，但可能仍在启动中。")


def main() -> None:
    root = Path(__file__).resolve().parent
    os.chdir(root)

    print("========================================")
    print("  电商用户画像系统 - 一键启动")
    print("========================================")
    print()

    # 1. 安装依赖（多次执行也没问题，已安装的包会被跳过或快速检查）
    print("[1/4] 安装依赖（如果已安装会自动跳过）...")
    run_step(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        "pip 安装 requirements.txt",
    )
    print()

    # 2. 可选：检测本地 CSV 并导入 MySQL（如果没有 CSV 就跳过此步）
    csv_path = root / "customer_features_rfmbc.csv"
    if csv_path.exists():
        print("[2/4] 检测到 customer_features_rfmbc.csv，尝试导入 MySQL ...")
        run_step(
            [sys.executable, "import_csv_to_mysql.py"],
            "导入 CSV 到 MySQL (import_csv_to_mysql.py)",
        )
    else:
        print(
            "[2/4] 未找到 customer_features_rfmbc.csv，跳过导入步骤（Flask 将直接使用 CSV 或当前数据源配置）。"
        )
    print()

    # 3. 启动 Flask 应用（默认开启 USE_MYSQL=1，让应用优先使用数据库）
    print("[3/4] 启动 Flask 服务 (flask_app.py) ...")
    env = os.environ.copy()
    # 默认启用 MySQL，如需仅使用 CSV，可在终端中取消该环境变量
    env.setdefault("USE_MYSQL", "1")
    try:
        flask_proc = subprocess.Popen(
            [sys.executable, "flask_app.py"],
            env=env,
        )
    except Exception as e:
        print(f"  ❌ 启动 Flask 失败：{e}")
        return

    # 4. 等待服务就绪并自动在默认浏览器中打开首页
    print()
    print("[4/4] 等待服务启动，并打开浏览器 ...")
    wait_for_server("http://127.0.0.1:5000", timeout=20.0)
    webbrowser.open("http://127.0.0.1:5000")

    print()
    print("✅ 所有进程已启动，Flask 正在后台运行。")
    print("   您可以在浏览器中访问：http://127.0.0.1:5000")
    print()
    print("提示：关闭 Flask 服务可在其终端窗口中按 Ctrl+C。")


if __name__ == "__main__":
    main()

