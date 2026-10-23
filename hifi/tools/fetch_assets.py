"""
fetch_assets.py — Downloads the CC0 Poly Haven assets used by the hi-fi Godot renderer.

Reads hifi/assets_manifest.json plus any hifi/godot/envs/<env>/assets.json, and writes:
  hifi/godot/assets/polyhaven/hdris/<name>_<res>.hdr
  hifi/godot/assets/polyhaven/textures/<name>/<name>_{diff,nor_gl,arm|rough}_<res>.jpg
  hifi/godot/assets/polyhaven/models/<name>/<name>.glb      (glTF + textures packed into one GLB)
  hifi/godot/assets/polyhaven/index.json                    (what is available, read by Godot)

The assets folder carries a .gdignore, so the Godot editor never imports it; the renderer loads
the files at runtime (Image.load_from_file / GLTFDocument). Everything is CC0 (polyhaven.com).

Usage (from the repo root):
  python hifi/tools/fetch_assets.py             # fetch everything missing
  python hifi/tools/fetch_assets.py --dry-run   # list download sizes only
  python hifi/tools/fetch_assets.py --names snow_02 boulder_01
  python hifi/tools/fetch_assets.py --index-only  # rebuild index.json (incl. model triangle counts)
"""
import argparse
import glob
import hashlib
import json
import os
import struct
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HIFI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HIFI, "godot", "assets", "polyhaven")
API = "https://api.polyhaven.com/files/"
UA = {"User-Agent": "orthoworlds-hifi-fetch/1.0"}
TEX_MAPS = [("diff", ["Diffuse"]), ("nor_gl", ["nor_gl"]), ("arm", ["arm"])]


def load_manifest():
    with open(os.path.join(HIFI, "assets_manifest.json")) as f:
        man = json.load(f)
    for path in sorted(glob.glob(os.path.join(HIFI, "godot", "envs", "*", "assets.json"))):
        with open(path) as f:
            extra = json.load(f)
        for kind in ("hdris", "textures", "models"):
            man.setdefault(kind, {}).update(extra.get(kind, {}))
    return man


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def download(url, dest, md5=None):
    """Downloads url to dest unless a file with the right md5 is already there. Returns bytes written."""
    if os.path.exists(dest) and (md5 is None or _md5(dest) == md5):
        return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    tmp = dest + ".part"
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    if md5 is not None and _md5(tmp) != md5:
        os.remove(tmp)
        raise IOError(f"md5 mismatch for {url}")
    os.replace(tmp, dest)
    return os.path.getsize(dest)


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------- plans
def plan_hdri(name, res):
    info = get_json(API + name)["hdri"][res]["hdr"]
    return [(info["url"], os.path.join(OUT, "hdris", f"{name}_{res}.hdr"), info.get("md5"), info["size"])]


def plan_texture(name, res):
    files = get_json(API + name)
    jobs = []
    for short, keys in TEX_MAPS:
        found = None
        for k in keys:
            if k in files and res in files[k]:
                fmts = files[k][res]
                found = fmts.get("jpg") or fmts.get("png")
                break
        if found is None and short == "arm" and "Rough" in files:  # no packed ARM: use roughness alone
            fmts = files["Rough"][res]
            found, short = fmts.get("jpg") or fmts.get("png"), "rough"
        if found is None:
            continue
        ext = os.path.splitext(found["url"])[1]
        jobs.append((found["url"], os.path.join(OUT, "textures", name, f"{name}_{short}_{res}{ext}"),
                     found.get("md5"), found["size"]))
    return jobs


def plan_model(name, res):
    g = get_json(API + name)["gltf"][res]["gltf"]
    size = g["size"] + sum(v["size"] for v in g.get("include", {}).values())
    return g, size


def pack_glb(gltf_path, glb_path):
    """Packs a .gltf with external .bin/images into a single .glb (images embedded as bufferViews)."""
    base = os.path.dirname(gltf_path)
    with open(gltf_path) as f:
        doc = json.load(f)
    blob = bytearray()
    buffer_offsets = []
    for buf in doc.get("buffers", []):
        while len(blob) % 4:
            blob.append(0)
        buffer_offsets.append(len(blob))
        with open(os.path.join(base, buf["uri"]), "rb") as f:
            blob += f.read()
    for view in doc.get("bufferViews", []):
        view["byteOffset"] = view.get("byteOffset", 0) + buffer_offsets[view.get("buffer", 0)]
        view["buffer"] = 0
    for img in doc.get("images", []):
        uri = img.pop("uri", None)
        if uri is None:
            continue
        with open(os.path.join(base, urllib.request.url2pathname(uri)), "rb") as f:
            data = f.read()
        while len(blob) % 4:
            blob.append(0)
        doc.setdefault("bufferViews", []).append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
        img["bufferView"] = len(doc["bufferViews"]) - 1
        img["mimeType"] = "image/png" if uri.lower().endswith(".png") else "image/jpeg"
        blob += data
    while len(blob) % 4:
        blob.append(0)
    doc["buffers"] = [{"byteLength": len(blob)}]
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    os.makedirs(os.path.dirname(glb_path), exist_ok=True)
    with open(glb_path + ".part", "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(js), b"JSON") + js)
        f.write(struct.pack("<I4s", len(blob), b"BIN\x00") + bytes(blob))
    os.replace(glb_path + ".part", glb_path)


def fetch_model(name, g):
    glb = os.path.join(OUT, "models", name, f"{name}.glb")
    if os.path.exists(glb):
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        gl = os.path.join(tmp, f"{name}.gltf")
        download(g["url"], gl, g.get("md5"))
        for rel, info in g.get("include", {}).items():
            download(info["url"], os.path.join(tmp, rel), info.get("md5"))
        pack_glb(gl, glb)
    return os.path.getsize(glb)


def glb_triangles(path):
    """Triangle count of a GLB (sum over mesh primitives; heavy scans are 30k-17M)."""
    with open(path, "rb") as f:
        _, _, _ = struct.unpack("<4sII", f.read(12))
        n, _ = struct.unpack("<I4s", f.read(8))
        doc = json.loads(f.read(n))
    acc = doc.get("accessors", [])
    tris = 0
    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if prim.get("mode", 4) != 4:
                continue
            if "indices" in prim:
                tris += acc[prim["indices"]]["count"] // 3
            elif "POSITION" in prim.get("attributes", {}):
                tris += acc[prim["attributes"]["POSITION"]]["count"] // 3
    return tris


def write_index():
    idx = {"hdris": {}, "textures": {}, "models": {}, "model_tris": {}}
    for p in sorted(glob.glob(os.path.join(OUT, "hdris", "*.hdr"))):
        stem = os.path.basename(p)[:-4]
        idx["hdris"][stem.rsplit("_", 1)[0]] = "res://assets/polyhaven/hdris/" + os.path.basename(p)
    for d in sorted(glob.glob(os.path.join(OUT, "textures", "*"))):
        name = os.path.basename(d)
        maps = {}
        for p in sorted(glob.glob(os.path.join(d, "*"))):
            fn = os.path.basename(p)
            if fn.endswith(".part"):
                continue
            short = fn[len(name) + 1:].rsplit("_", 1)[0]
            maps[short] = f"res://assets/polyhaven/textures/{name}/{fn}"
        if "diff" in maps:
            idx["textures"][name] = maps
    for p in sorted(glob.glob(os.path.join(OUT, "models", "*", "*.glb"))):
        name = os.path.basename(p)[:-4]
        idx["models"][name] = "res://assets/polyhaven/models/%s/%s" % (os.path.basename(os.path.dirname(p)), os.path.basename(p))
        try:
            idx["model_tris"][name] = glb_triangles(p)
        except (OSError, ValueError, KeyError, struct.error):
            pass
    tmp = os.path.join(OUT, f"index.json.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(idx, f, indent=1)
    os.replace(tmp, os.path.join(OUT, "index.json"))  # atomic: several fetches may run at once
    return idx


def main(args):
    if args.index_only:
        idx = write_index()
        heavy = sorted(idx["model_tris"].items(), key=lambda kv: -kv[1])[:8]
        print("index rebuilt; heaviest models:", ", ".join(f"{n} {t // 1000}k" for n, t in heavy))
        return 0
    man = load_manifest()
    os.makedirs(OUT, exist_ok=True)
    gdignore = os.path.join(HIFI, "godot", "assets", ".gdignore")
    if not os.path.exists(gdignore):
        open(gdignore, "w").close()

    def wanted(name):
        return not args.names or name in args.names

    jobs, models, failures = [], [], []

    def plan(kind, name, res):
        try:
            if kind == "hdris":
                return kind, name, plan_hdri(name, res)
            if kind == "textures":
                return kind, name, plan_texture(name, res)
            return kind, name, plan_model(name, res)
        except Exception as e:
            return kind, name, e

    todo = [(k, n, r) for k in ("hdris", "textures", "models") for n, r in man.get(k, {}).items() if wanted(n)]
    with ThreadPoolExecutor(8) as pool:
        for kind, name, res in pool.map(lambda t: plan(*t), todo):
            if isinstance(res, Exception):
                failures.append((name, res))
            elif kind == "models":
                models.append((name, res[0], res[1]))
            else:
                jobs.extend((name, *j) for j in res)

    total = sum(j[4] for j in jobs) + sum(m[2] for m in models)
    print(f"{len(jobs)} files + {len(models)} models, {total / 1e6:.1f} MB total")
    if args.dry_run:
        sizes = {}
        for name, _, _, _, size in jobs:
            sizes[name] = sizes.get(name, 0) + size
        for name, _, size in models:
            sizes[name] = size
        for name, size in sorted(sizes.items(), key=lambda kv: -kv[1]):
            print(f"  {size / 1e6:7.2f} MB  {name}")
    else:
        def run_file(j):
            name, url, dest, md5, _ = j
            try:
                return download(url, dest, md5)
            except Exception as e:
                failures.append((name, e))
                return 0

        def run_model(m):
            try:
                return fetch_model(m[0], m[1])
            except Exception as e:
                failures.append((m[0], e))
                return 0

        with ThreadPoolExecutor(8) as pool:
            got = sum(pool.map(run_file, jobs)) + sum(pool.map(run_model, models))
        idx = write_index()
        print(f"downloaded {got / 1e6:.1f} MB; available: {len(idx['hdris'])} HDRIs, "
              f"{len(idx['textures'])} textures, {len(idx['models'])} models -> {OUT}")
    for name, e in failures:
        print(f"  [FAIL] {name}: {e}")
    return 1 if failures else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fetch CC0 Poly Haven assets for the hi-fi Godot renderer")
    p.add_argument("--dry-run", action="store_true", help="Only list what would be downloaded")
    p.add_argument("--names", nargs="*", default=None, help="Only these asset names")
    p.add_argument("--index-only", action="store_true", help="Only rebuild index.json")
    sys.exit(main(p.parse_args()))
