"""
Utilities for on-the-fly video frame extraction and side-by-side overlay creation.

Supports two source types:
- MP4 files: decoded with `decord` (efficient random access)
- PNG directories: fallback for pre-extracted frames (frame_000000.png, etc.)
"""

import hashlib
import os
import tempfile
from functools import lru_cache

from PIL import Image

_S3_VIDEO_CACHE_DIR = os.path.join(tempfile.gettempdir(), "video_s3_cache")


@lru_cache(maxsize=256)
def _get_video_reader(video_path: str):
    """Cache decord.VideoReader objects to avoid re-opening files."""
    import decord
    return decord.VideoReader(video_path, num_threads=1)


def extract_frame(source_path: str, frame_idx: int) -> Image.Image:
    """
    Extract a single frame as a PIL Image.

    Args:
        source_path: Either an MP4 file path or a directory of PNGs.
        frame_idx: The frame index to extract.

    Returns:
        PIL Image in RGB mode.
    """
    if source_path.endswith(".mp4"):
        vr = _get_video_reader(source_path)
        frame = vr[frame_idx]
        # decord returns NDArray regardless of bridge; .asnumpy() always works
        frame_array = frame.asnumpy()
        return Image.fromarray(frame_array)
    else:
        png_path = os.path.join(source_path, f"frame_{frame_idx:06d}.png")
        return Image.open(png_path).convert("RGB")


def create_side_by_side(img1: Image.Image, img2: Image.Image) -> Image.Image:
    """
    Create a side-by-side image with a yellow separator line.

    Args:
        img1: Left image.
        img2: Right image (resized to match img1 if needed).

    Returns:
        Combined PIL Image (width*2 + 2, height).
    """
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.LANCZOS)
    w, h = img1.size
    combined = Image.new("RGB", (w * 2 + 2, h))
    combined.paste(img1, (0, 0))
    combined.paste(Image.new("RGB", (2, h), (255, 255, 0)), (w, 0))
    combined.paste(img2, (w + 2, 0))
    return combined


def get_frame_source(video_path: str) -> str:
    """
    Determine the best source for frames: MP4 file or PNG directory.

    Args:
        video_path: Path to the MP4 file (as stored in eval_log.json).

    Returns:
        The MP4 path if the file exists, otherwise the PNG directory path.

    Raises:
        FileNotFoundError: If neither the MP4 nor the PNG directory exists.
    """
    if os.path.isfile(video_path):
        return video_path
    png_dir = video_path[:-4]  # strip .mp4
    if os.path.isdir(png_dir):
        return png_dir
    raise FileNotFoundError(f"Neither {video_path} nor {png_dir} found")

import os
import glob
import boto3
from urllib.parse import urlparse

s3 = boto3.client("s3")

def parse_s3_uri(uri: str):
    # uri like s3://bucket/some/prefix
    u = urlparse(uri)
    bucket = u.netloc
    prefix = u.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix

def s3_list_immediate_subprefixes(bucket: str, prefix: str):
    """
    Lists "directories" immediately under prefix using Delimiter='/'
    Returns list of prefixes like 'some/prefix/job123/'
    """
    subprefixes = []
    token = None
    while True:
        kwargs = dict(Bucket=bucket, Prefix=prefix, Delimiter="/")
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)

        for cp in resp.get("CommonPrefixes", []):
            subprefixes.append(cp["Prefix"])

        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return subprefixes

def s3_key_exists(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False

def read_s3_json(uri: str):
    """Read a JSON file from S3 and return the parsed object."""
    import json
    bucket, prefix = parse_s3_uri(uri)
    # parse_s3_uri adds trailing '/', strip it for a file key
    key = prefix.rstrip("/")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def find_job_dirs(task_path: str):
    if task_path.startswith("s3://"):
        bucket, prefix = parse_s3_uri(task_path)
        subdirs = s3_list_immediate_subprefixes(bucket, prefix)

        # Keep only those with eval_log.json
        job_prefixes = []
        for p in subdirs:
            if s3_key_exists(bucket, p + "eval_log.json"):
                job_prefixes.append(f"s3://{bucket}/{p}")
        return sorted(job_prefixes)

    # Local filesystem fallback
    all_dirs = sorted(glob.glob(f"{task_path}/*"))
    job_dirs = [
        d for d in all_dirs
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "eval_log.json"))
    ]
    return job_dirs
