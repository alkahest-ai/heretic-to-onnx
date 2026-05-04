#!/usr/bin/env python3
"""Stream Kaggle kernel output files to disk.

The Kaggle CLI currently downloads each output file into memory before writing
it. That is fragile for multi-GB model shards, so this helper uses the same
kernel output API and writes response chunks directly to disk.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", help="Kernel slug in owner/kernel form.")
    parser.add_argument("--path", required=True, help="Directory to download into.")
    parser.add_argument("--file-pattern", required=True, help="Regex matched against Kaggle output paths.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024 * 16)
    parser.add_argument("--retries", type=int, default=5, help="Retry count for interrupted downloads.")
    return parser


def _list_outputs(api: KaggleApi, owner: str, slug: str):
    with api.build_kaggle_client() as kaggle:
        request = ApiListKernelSessionOutputRequest()
        request.user_name = owner
        request.kernel_slug = slug
        return kaggle.kernels.kernels_api_client.list_kernel_session_output(request)


def _content_range_total(value: str | None) -> int | None:
    if not value or "/" not in value:
        return None
    total = value.rsplit("/", 1)[-1]
    if total == "*":
        return None
    try:
        return int(total)
    except ValueError:
        return None


def _download(url: str, output_path: Path, *, chunk_size: int, force: bool, retries: int) -> None:
    if output_path.exists() and not force:
        print(f"exists: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.with_name(output_path.name + ".part")
    if part_path.exists() and force:
        part_path.unlink()

    attempt = 0
    while True:
        start = part_path.stat().st_size if part_path.exists() else 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        try:
            with requests.get(url, stream=True, timeout=(30, 600), headers=headers) as response:
                if start and response.status_code != 206:
                    part_path.unlink(missing_ok=True)
                    start = 0
                    response.close()
                    continue
                response.raise_for_status()
                mode = "ab" if start else "wb"
                with part_path.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            handle.write(chunk)
                total = _content_range_total(response.headers.get("content-range"))
                if total is not None and part_path.stat().st_size < total:
                    raise requests.exceptions.ChunkedEncodingError(
                        f"incomplete range download: {part_path.stat().st_size} < {total}"
                    )
                break
        except requests.RequestException as exc:
            attempt += 1
            if attempt > retries:
                raise
            print(f"retrying: {output_path} attempt={attempt}/{retries} error={type(exc).__name__}: {exc}")
            time.sleep(min(30, 2**attempt))

    os.replace(part_path, output_path)
    print(f"downloaded: {output_path} ({output_path.stat().st_size} bytes)")


def main() -> int:
    args = _parser().parse_args()
    owner, slug = args.kernel.split("/", 1)
    target_dir = Path(args.path).expanduser().resolve()
    pattern = re.compile(args.file_pattern)

    api = KaggleApi()
    api.authenticate()
    response = _list_outputs(api, owner, slug)
    matched = [item for item in response.files or [] if pattern.search(item.file_name)]
    if not matched:
        raise SystemExit(f"no files matched {args.file_pattern!r}")

    for item in matched:
        _download(
            item.url,
            target_dir / item.file_name,
            chunk_size=args.chunk_size,
            force=args.force,
            retries=args.retries,
        )
    if response.log:
        log_path = target_dir / f"{slug}.log"
        log_path.write_text(response.log, encoding="utf-8")
        print(f"downloaded: {log_path} ({log_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
