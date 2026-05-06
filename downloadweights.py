import os, re, sys, json, pathlib
import requests

REPO_ID = "Raymond-Qiancx/ProgressLM-3B-SFT"
LOCAL_DIR = "outputs/ProgressLM-3B-SFT"

# Read OIDs from pointer files
def parse_lfs_pointer(path):
    txt = pathlib.Path(path).read_text()
    m = re.search(r"oid sha256:([0-9a-f]{64})", txt)
    if not m:
        raise RuntimeError(f"Not an LFS pointer? {path}")
    oid = m.group(1)
    m2 = re.search(r"size (\d+)", txt)
    size = int(m2.group(1)) if m2 else None
    return oid, size

p1 = os.path.join(LOCAL_DIR, "model-00001-of-00002.safetensors")
p2 = os.path.join(LOCAL_DIR, "model-00002-of-00002.safetensors")

oids = []
for p in [p1, p2]:
    oid, size = parse_lfs_pointer(p)
    oids.append((p, oid, size))

print("Found OIDs:")
for p, oid, size in oids:
    print(" ", os.path.basename(p), oid, "size=", size)

# Use Hugging Face resolve endpoint (works even without git-lfs)
# This will 302 redirect to a storage URL and requests will stream it.
def download_file(filename, out_path):
    url = f"https://huggingface.co/{REPO_ID}/resolve/main/{filename}"
    print(f"\nDownloading {filename} ...")
    with requests.get(url, stream=True, allow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        tmp = out_path + ".tmp"
        total = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
                    if total % (1024*1024*512) < 1024*1024:  # every ~512MB
                        print(f"  downloaded {total/1e9:.2f} GB", flush=True)
        os.replace(tmp, out_path)
    print(f"Saved -> {out_path} ({total/1e9:.2f} GB)")

# IMPORTANT:
# If your environment requires HF auth, set:
#   export HF_TOKEN=...
# and we’ll pass it as a Bearer token.
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if token:
    requests.sessions.Session().headers.update({"Authorization": f"Bearer {token}"})
    print("Using HF token from env.")

# Download the actual files by filename (not by OID)
download_file("model-00001-of-00002.safetensors", p1)
download_file("model-00002-of-00002.safetensors", p2)

print("\nDone. Verify:")
print("  head -n 1", p1)
print("  should NOT say 'version https://git-lfs.github.com/spec/v1'")