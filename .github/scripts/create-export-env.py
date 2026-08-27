# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""根据导出器注册表构建并通过冒烟测试验证导出测试虚拟环境。"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ultralytics.engine.exporter import EXPORT_ENVS  # noqa: E402, RUF100


def isolated_env_ids():
    """返回应在共享基础 CI 环境之外运行的导出环境。"""
    return sorted(
        (env for env, recipe in EXPORT_ENVS.items() if recipe["python"]), key=lambda env: env != "isolated-deepx"
    )


def build_env(env_id, root):
    """构建一个导出环境并运行其中的导出冒烟测试命令。"""
    recipe = EXPORT_ENVS[env_id]
    venv = root / env_id

    shutil.rmtree(venv, ignore_errors=True)
    subprocess.run(["uv", "venv", str(venv), "--python", recipe["python"], "--seed"], check=True)
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    package = f"{REPO}[{','.join(recipe['extras'])}]"
    indexes = [token for flag, url in recipe["indexes"] for token in (flag, url)]
    index_strategy = ["--index-strategy", "unsafe-best-match"] if indexes else []
    torch = [f"torch{recipe['torch']}"] if recipe["torch"] else []
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "-e",
            package,
            "pytest",
            *torch,
            *recipe["requirements"],
            *indexes,
            *index_strategy,
            "--torch-backend",
            "cpu",
        ],
        check=True,
    )

    if recipe["env"]:
        site_packages = next(venv.glob("lib/python*/site-packages"))
        site_packages.joinpath("sitecustomize.py").write_text(
            "\n".join(f'import os; os.environ.setdefault("{k}", "{v}")' for k, v in recipe["env"].items()) + "\n"
        )

    env = {**os.environ, "YOLO_AUTOINSTALL": "false", **recipe["env"]}
    # 某些厂商转换器会直接调用 `pip`；将这些子进程限制在此隔离虚拟环境中。
    env["PATH"] = f"{python.parent}{os.pathsep}{env['PATH']}"
    yolo = venv / ("Scripts/yolo.exe" if os.name == "nt" else "bin/yolo")
    for command in recipe["smoke"]:
        cmd = command.split()
        executable = yolo if cmd[0] == "yolo" else venv / "bin" / cmd[0]
        subprocess.run([str(executable), *cmd[1:]], check=True, cwd=venv, env=env)


def main():
    """解析参数并构建请求的导出环境。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", action="append", choices=isolated_env_ids())
    parser.add_argument("--list", action="store_true", help="Print isolated export environment ids and exit.")
    args = parser.parse_args()
    envs = args.env or isolated_env_ids()

    if args.list:
        print("\n".join(envs))
        return

    root = Path(os.environ.get("ULTRALYTICS_ISOLATED_VENVS", "/opt/venvs"))
    root.mkdir(parents=True, exist_ok=True)
    for env_id in envs:
        build_env(env_id, root)


if __name__ == "__main__":
    main()
