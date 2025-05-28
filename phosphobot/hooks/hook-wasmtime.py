# phosphobot/hooks/hook-wasmtime.py
from PyInstaller.utils.hooks import collect_data_files, get_package_paths
import os

_package_name = "wasmtime"

datas = []

try:
    # Verify the package exists
    pkg_base_dirs = get_package_paths(_package_name)
    if not pkg_base_dirs:
        raise ImportError(f"Package '{_package_name}' not found by PyInstaller.")

    # Collect all data files from the root of the 'wasmtime' package.
    # Destination paths will be like 'wasmtime/filename'.
    collected_files = collect_data_files(
        package=_package_name,
        include_py_files=False,  # We don't want .py/.pyc files, PyInstaller handles them
    )

    if collected_files:
        datas.extend(collected_files)
        print(f"HOOK wasmtime: Bundled data from '{_package_name}': {collected_files}")
    else:
        expected_source_path = "unknown"
        if pkg_base_dirs:
            expected_source_path = os.path.join(
                pkg_base_dirs[0], _package_name.replace(".", os.sep)
            )
        print(
            f"HOOK INFO: No data files found by collect_data_files for '{_package_name}'. Searched near: {expected_source_path}"
        )

except ImportError as e:
    print(f"HOOK ERROR (wasmtime): {e}. Cannot collect data files.")
except Exception as e:
    print(f"HOOK UNEXPECTED ERROR (wasmtime): {e}")
