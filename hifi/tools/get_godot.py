"""
get_godot.py — Downloads an official Godot build into hifi/tools/bin (gitignored).

Under WSL2 the Linux build only gets software rendering (llvmpipe); the Windows build uses the
GPU with the Forward+ renderer and is launched from WSL by hifi/renderer.py, which looks in
hifi/tools/bin first.

  python hifi/tools/get_godot.py                 # Windows build (default under WSL / Windows)
  python hifi/tools/get_godot.py --platform linux
"""
import argparse
import io
import os
import stat
import sys
import urllib.request
import zipfile

BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
URL = "https://github.com/godotengine/godot/releases/download/{v}-stable/Godot_v{v}-stable_{p}.zip"
PLATFORMS = {"windows": "win64.exe", "linux": "linux.x86_64"}


def main(args):
    os.makedirs(BIN, exist_ok=True)
    url = URL.format(v=args.version, p=PLATFORMS[args.platform])
    print(f"downloading {url} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "orthoworlds-hifi"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for member in z.namelist():
            out = os.path.join(BIN, os.path.basename(member))
            with z.open(member) as src, open(out, "wb") as dst:
                dst.write(src.read())
            os.chmod(out, os.stat(out).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            print("  ", out)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download a Godot build into hifi/tools/bin")
    p.add_argument("--version", default="4.3")
    p.add_argument("--platform", choices=sorted(PLATFORMS), default="windows")
    sys.exit(main(p.parse_args()))
