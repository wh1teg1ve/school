"""
一键启动电商用户画像系统（Python 版本）

步骤：
1. 安装依赖（requirements.txt）
2. 如设置 USE_MYSQL=1：可选导入到 MySQL（import_csv_to_mysql.py）
3. 启动 Flask（flask_app.py，默认使用 CSV）
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
            print(f"  [警告] 命令执行返回码 {result.returncode}，可手动检查：{' '.join(cmd)}")
    except Exception as e:
        print(f"  [警告] 命令执行失败：{e}")


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
    print("  [警告] 在预期时间内未检测到 Flask 服务，但可能仍在启动中。")


def _free_port_5000_on_windows() -> None:
    """Windows 下释放 5000 端口，避免旧 Flask 进程占用导致新进程启动失败。"""
    if os.name != "nt":
        return
    try:
        # netstat 输出示例：TCP    127.0.0.1:5000   0.0.0.0:0   LISTENING   12345
        out = subprocess.check_output(["netstat", "-ano", "-p", "tcp"], text=True, errors="ignore")
    except Exception:
        return

    pids: set[int] = set()
    for line in out.splitlines():
        if "LISTENING" not in line:
            continue
        if ":5000" not in line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        pid_str = parts[-1]
        try:
            pids.add(int(pid_str))
        except Exception:
            continue

    for pid in sorted(pids):
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


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

    # 2. 可选：仅当你显式启用 USE_MYSQL=1 时才导入 MySQL
    use_mysql = os.environ.get("USE_MYSQL", "").lower() in ("1", "true", "yes")
    csv_path = root / "customer_features_rfmbc.csv"
    if use_mysql and csv_path.exists():
        print("[2/4] 已启用 USE_MYSQL=1，检测到 customer_features_rfmbc.csv，尝试导入 MySQL ...")
        run_step(
            [sys.executable, "import_csv_to_mysql.py"],
            "导入 CSV 到 MySQL (import_csv_to_mysql.py)",
        )
    elif use_mysql and (not csv_path.exists()):
        print("[2/4] 已启用 USE_MYSQL=1，但未找到 customer_features_rfmbc.csv，跳过导入步骤。")
    else:
        print("[2/4] 未启用 USE_MYSQL（默认使用 CSV），跳过 MySQL 导入步骤。")
    print()

    # 3. 启动 Flask 应用（默认不强制 USE_MYSQL，保持 CSV 模式可直接运行）
    print("[3/4] 启动 Flask 服务 (flask_app.py) ...")
    env = os.environ.copy()
    # 若用户设置了 USE_MYSQL=1，则沿用；否则保持不设置（走 CSV）
    try:
        _free_port_5000_on_windows()
        flask_proc = subprocess.Popen(
            [sys.executable, "flask_app.py"],
            env=env,
        )
    except Exception as e:
        print(f"  [错误] 启动 Flask 失败：{e}")
        return

    # 4. 等待服务就绪并自动在默认浏览器中打开首页
    print()
    print("[4/4] 等待服务启动，并打开浏览器 ...")
    wait_for_server("http://127.0.0.1:5000", timeout=20.0)
    webbrowser.open("http://127.0.0.1:5000")

    print()
    print("[完成] 所有进程已启动，Flask 正在后台运行。")
    print("   您可以在浏览器中访问：http://127.0.0.1:5000")
    print()
    print("提示：关闭 Flask 服务可在其终端窗口中按 Ctrl+C。")


if __name__ == "__main__":
    main()

