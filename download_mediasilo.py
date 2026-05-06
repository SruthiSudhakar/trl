#!/usr/bin/env python3
"""
Download all videos from a MediaSilo spotlight page, organized by subfolder.

Usage:
    python3 download_mediasilo.py [--output-dir OUTPUT_DIR] [--explore-only]
"""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright

SPOTLIGHT_URL = "https://app.mediasilo.com/spotlight/21e8c21f-fd7a-47d9-91bc-67cb987303f4"
SPOTLIGHT_ID = "21e8c21f-fd7a-47d9-91bc-67cb987303f4"
PLAYLIST_ID = "2dc2bb7e-82e5-4d41-a768-302da5d1f95a"
API_BASE = "https://api.mediasilo.com"


def sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', '_', name).strip()


def download_file(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"    [SKIP] Already exists: {dest.name}")
        return
    print(f"    [DL] {dest.name} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as response:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"    [OK] {size_mb:.1f} MB")
    except Exception as e:
        print(f"    [ERROR] {dest.name}: {e}")
        if dest.exists():
            dest.unlink()


def api_fetch(page, path, params=""):
    """Fetch from MediaSilo API using the page's browser context."""
    url = f"{API_BASE}{path}"
    if params:
        url += f"?{params}"
    result = page.evaluate("""async (url) => {
        try {
            const resp = await fetch(url);
            const text = await resp.text();
            return {status: resp.status, body: text};
        } catch(e) {
            return {error: e.message};
        }
    }""", url)
    if result.get("error"):
        return None, result["error"]
    try:
        body = json.loads(result["body"])
    except json.JSONDecodeError:
        body = result["body"]
    return result["status"], body


def get_download_url(asset):
    """Extract the best download URL from an asset dict."""
    # Check direct URL fields
    for key in ["sourceUrl", "proxyUrl", "downloadUrl", "url"]:
        if asset.get(key):
            return asset[key]

    # Check nested proxy/source objects
    proxy = asset.get("proxy")
    if isinstance(proxy, dict):
        for key in ["url", "downloadUrl", "sourceUrl"]:
            if proxy.get(key):
                return proxy[key]

    source = asset.get("source")
    if isinstance(source, dict):
        for key in ["url", "downloadUrl", "sourceUrl"]:
            if source.get(key):
                return source[key]

    # Check derivatives
    derivs = asset.get("derivatives") or asset.get("transcodes") or []
    if isinstance(derivs, list):
        for d in derivs:
            if isinstance(d, dict) and d.get("url"):
                return d["url"]

    return None


def main():
    parser = argparse.ArgumentParser(description="Download MediaSilo spotlight videos")
    parser.add_argument("--output-dir", default="mediasilo_videos", help="Output directory")
    parser.add_argument("--explore-only", action="store_true", help="Only explore, don't download")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        )
        page = context.new_page()

        print("[1/3] Loading spotlight page...")
        page.goto(SPOTLIGHT_URL, wait_until="networkidle", timeout=60000)
        time.sleep(3)

        # Fetch folder list
        print("[2/3] Fetching folders and assets...")
        status, folders = api_fetch(
            page,
            f"/v3/presentations/{SPOTLIGHT_ID}/playlists/{PLAYLIST_ID}/folders",
            "_pageSize=100"
        )
        if status != 200:
            print(f"  ERROR: Could not fetch folders (status={status})")
            browser.close()
            return

        print(f"  Found {len(folders)} folders")

        # First, let's dump one full asset to see all available fields
        first_folder = folders[0]
        status, sample_assets = api_fetch(
            page,
            f"/v3/presentations/{SPOTLIGHT_ID}/playlists/{PLAYLIST_ID}/folders/{first_folder['id']}/assets",
            "_pageSize=1"
        )
        if isinstance(sample_assets, list) and sample_assets:
            print(f"\n  Sample asset fields: {list(sample_assets[0].keys())}")
            print(f"  Full sample asset:\n{json.dumps(sample_assets[0], indent=2)[:2000]}")

        # Now fetch all assets for all folders
        all_assets = {}
        total_count = 0

        for folder in folders:
            folder_name = folder["name"]
            folder_id = folder["id"]
            print(f"\n  [{folder_name}]")

            folder_assets = []
            page_num = 0
            page_size = 100

            while True:
                status, assets = api_fetch(
                    page,
                    f"/v3/presentations/{SPOTLIGHT_ID}/playlists/{PLAYLIST_ID}/folders/{folder_id}/assets",
                    f"_pageSize={page_size}&_page={page_num}"
                )

                if status != 200 or not assets or not isinstance(assets, list):
                    break

                folder_assets.extend(assets)
                if len(assets) < page_size:
                    break
                page_num += 1

            # Count how many have download URLs
            with_urls = sum(1 for a in folder_assets if get_download_url(a))
            print(f"    {len(folder_assets)} assets ({with_urls} with download URLs)")

            all_assets[folder_name] = folder_assets
            total_count += len(folder_assets)

        print(f"\n  Total: {total_count} assets across {len(folders)} folders")

        # If no download URLs found, try to get them individually
        sample = all_assets[folders[0]["name"]][0] if all_assets else {}
        if not get_download_url(sample):
            print("\n  No direct download URLs in list response.")
            print("  Trying to fetch individual asset details...")

            asset_id = sample.get("id", "")
            if asset_id:
                # Try various endpoints to get individual asset with download URL
                endpoints = [
                    f"/v3/presentations/{SPOTLIGHT_ID}/assets/{asset_id}",
                    f"/v3/presentations/{SPOTLIGHT_ID}/playlists/{PLAYLIST_ID}/assets/{asset_id}",
                    f"/v3/assets/{asset_id}?presentationId={SPOTLIGHT_ID}",
                    f"/v3/presentations/{SPOTLIGHT_ID}/assets/{asset_id}/source",
                    f"/v3/presentations/{SPOTLIGHT_ID}/assets/{asset_id}/proxy",
                ]
                for ep in endpoints:
                    status, detail = api_fetch(page, ep)
                    print(f"    [{status}] {ep}")
                    if isinstance(detail, dict):
                        detail_str = json.dumps(detail)[:500]
                        print(f"         {detail_str}")
                    elif isinstance(detail, str):
                        print(f"         {detail[:500]}")

        # Phase 3: Download using available URLs
        if args.explore_only:
            print("\n[3/3] Explore-only mode.")
            # Save full manifest with all data
            manifest_path = output_dir / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(all_assets, f, indent=2)
            print(f"  Full manifest saved to {manifest_path}")
        else:
            print(f"\n[3/3] Downloading videos...")
            downloaded = 0
            skipped = 0

            for folder_name, assets in all_assets.items():
                if not assets:
                    continue
                folder_dir = output_dir / sanitize_filename(folder_name)
                folder_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n  [{folder_name}] -> {folder_dir}")

                for i, asset in enumerate(assets):
                    title = asset.get("title") or asset.get("fileName") or f"video_{i}"
                    name = sanitize_filename(title)

                    url = get_download_url(asset)
                    if not url:
                        # Try fetching individual asset detail for URL
                        asset_id = asset.get("id", "")
                        if asset_id:
                            s, detail = api_fetch(
                                page,
                                f"/v3/presentations/{SPOTLIGHT_ID}/assets/{asset_id}/source"
                            )
                            if isinstance(detail, dict):
                                url = detail.get("url") or detail.get("downloadUrl")
                            if not url:
                                s, detail = api_fetch(
                                    page,
                                    f"/v3/presentations/{SPOTLIGHT_ID}/assets/{asset_id}/proxy"
                                )
                                if isinstance(detail, dict):
                                    url = detail.get("url") or detail.get("downloadUrl")

                    if not url:
                        skipped += 1
                        continue

                    if not any(name.lower().endswith(ext) for ext in (".mp4", ".mov", ".avi", ".mkv", ".webm")):
                        name += ".mp4"

                    download_file(url, folder_dir / name)
                    downloaded += 1

            print(f"\n  Downloaded: {downloaded}, Skipped (no URL): {skipped}")

        browser.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
