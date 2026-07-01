"""Statically verify a built mmcv wheel contains the compiled extension.

Confirms the CUDA/CPU native extension (``mmcv/_ext.<abi>.so`` or
``mmcv/_ext.<abi>.pyd``) actually compiled and got packaged into the wheel,
without importing/running it — CI runners have no GPU, so a CUDA build's _ext
must never be loaded here. Guards against silently shipping an extension-less
wheel (e.g. torch mis-detected as CPU, or MMCV_WITH_OPS=0).

Usage: python check_wheel_ext.py <dist_dir>
Exit 0 if exactly one wheel exists and it holds a compiled mmcv/_ext binary,
non-zero (with a message) otherwise.
"""

import glob
import os
import sys
import zipfile

EXT_SUFFIXES = (".so", ".pyd")


def main(dist_dir: str) -> int:
    wheels = sorted(glob.glob(os.path.join(dist_dir, "*.whl")))
    if len(wheels) != 1:
        print(f"ERROR: expected exactly one wheel in {dist_dir!r}, " f"found {wheels}")
        return 1

    wheel = wheels[0]
    names = zipfile.ZipFile(wheel).namelist()
    ext = [n for n in names if n.startswith("mmcv/_ext") and n.endswith(EXT_SUFFIXES)]

    print(f"wheel: {os.path.basename(wheel)}")
    if not ext:
        print(
            "ERROR: no compiled mmcv/_ext.(so|pyd) found in the wheel — "
            "the native extension was not built."
        )
        return 1

    for n in ext:
        print(f"  compiled extension: {n}")
    print("OK: compiled extension present in wheel.")
    return 0


if __name__ == "__main__":
    dist = sys.argv[1] if len(sys.argv) > 1 else "dist"
    raise SystemExit(main(dist))
