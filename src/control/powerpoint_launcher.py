from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _registry_powerpoint_path() -> Path | None:
    if os.name != "nt":
        return None
    try:
        import winreg

        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            try:
                with winreg.OpenKey(hive, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\powerpnt.exe") as key:
                    value, _ = winreg.QueryValueEx(key, "")
                    path = Path(value.strip('"'))
                    if path.is_file():
                        return path
            except OSError:
                continue
    except ImportError:
        return None
    return None


def find_powerpoint_executable() -> Path | None:
    path_result = shutil.which("POWERPNT.EXE")
    if path_result:
        return Path(path_result)

    registry_path = _registry_powerpoint_path()
    if registry_path is not None:
        return registry_path

    roots = {
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
        os.environ.get("ProgramW6432"),
    }
    office_versions = ("Office16", "Office15", "Office14")
    for root in filter(None, roots):
        office_root = Path(root) / "Microsoft Office"
        for version in office_versions:
            for relative in (Path("root") / version, Path(version)):
                candidate = office_root / relative / "POWERPNT.EXE"
                if candidate.is_file():
                    return candidate

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        alias = Path(local_app_data) / "Microsoft" / "WindowsApps" / "POWERPNT.EXE"
        if alias.is_file():
            return alias
    return None


def _is_powerpoint_running() -> bool:
    try:
        result = subprocess.run(
            ["tasklist.exe", "/FI", "IMAGENAME eq POWERPNT.EXE", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return "POWERPNT.EXE" in result.stdout.upper()
    except (OSError, subprocess.SubprocessError):
        return False


def launch_powerpoint() -> tuple[str, str]:
    if os.name != "nt":
        return "not_found", "Microsoft PowerPoint could not be located. Please open PowerPoint manually."

    if _is_powerpoint_running():
        return "launched", "PowerPoint is already running. Open your presentation and start Slide Show."

    executable = find_powerpoint_executable()
    if executable is None:
        return "not_found", "Microsoft PowerPoint could not be located. Please open PowerPoint manually."

    try:
        subprocess.Popen(
            [str(executable)],
            close_fds=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return "error", f"PowerPoint could not be launched: {exc}"
    return "launched", "PowerPoint launched. Open your presentation and start Slide Show."