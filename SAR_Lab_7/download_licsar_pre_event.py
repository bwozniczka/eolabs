#!/usr/bin/env python3
"""
Download pre-event COMET-LiCSAR interferograms for one frame.

Designed for cases where:

Directory listing is here:
    https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/62/062D_05831_131313/interferograms/

But actual files are linked from:
    https://data.ceda.ac.uk/neodc/comet/data/licsar_products/62/062D_05831_131313/20161227_20170120/...

Example test:
    python download_licsar_pre_event.py ^
      --base-url https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/62/062D_05831_131313/ ^
      --out C:\\sholtkamp\\licsar_test ^
      --test-one-png

Example full pre-event download:
    python download_licsar_pre_event.py ^
      --base-url https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/62/062D_05831_131313/ ^
      --out C:\\sholtkamp\\licsar_062D_05831_131313_pre_20170624 ^
      --cutoff 20170624 ^
      --max-days 72 ^
      --products unw cc png

Default:
    - Selects interferogram pairs whose END date is before cutoff.
    - Excludes pairs ending on the cutoff date itself.
    - Downloads unwrapped phase and coherence unless products are specified.
    - Downloads metadata / geometry rasters when available.

Notes:
    LiCSAR .geo.unw.tif is unwrapped phase, normally in radians.
    LiCSAR .geo.cc.tif is coherence.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse, unquote

import requests


PAIR_RE = re.compile(r"^(\d{8})_(\d{8})/?$")
FRAME_RE = re.compile(r"LiCSAR_products/([^/]+)/([^/]+)/?")
URL_RE = re.compile(r"https?://[^\s\"'<>]+")


class LinkParser(HTMLParser):
    """Extract hrefs from normal HTML directory listings."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs) -> None:
        if tag.lower() != "a":
            return
        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
        if href:
            self.links.append(href)


@dataclass(frozen=True)
class IfgPair:
    name: str
    start: datetime
    end: datetime
    listing_url: str

    @property
    def temporal_baseline_days(self) -> int:
        return (self.end - self.start).days


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 LiCSAR-student-downloader/2.0 "
                "(compatible; educational use)"
            )
        }
    )
    return session


def get_text(session: requests.Session, url: str, timeout: int = 60) -> str:
    response = session.get(url, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    return response.text


def list_hrefs(session: requests.Session, url: str) -> list[str]:
    """
    Return links from a directory listing.

    This is intentionally robust:
      1. Parse normal HTML hrefs.
      2. Also extract bare absolute URLs, because some listings appear as text.
      3. Also extract pair directory names and common filenames from plain text.
    """
    text = get_text(session, url)

    parser = LinkParser()
    parser.feed(text)

    links: list[str] = []

    # 1. HTML hrefs
    links.extend(parser.links)

    # 2. Bare absolute URLs in text
    links.extend(URL_RE.findall(text))

    # 3. Bare pair names or file names in text
    bare_tokens = re.findall(
        r"\b(?:\d{8}_\d{8}/?|[A-Za-z0-9_.-]+\.(?:tif|png|txt|csv)|baselines|network\.png)\b",
        text,
    )
    links.extend(bare_tokens)

    # De-duplicate while preserving order
    seen = set()
    clean_links: list[str] = []
    for link in links:
        link = link.strip()
        if not link or link in ("../", "..", "./", "."):
            continue
        if link not in seen:
            clean_links.append(link)
            seen.add(link)

    return clean_links


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d")


def derive_ceda_product_root(base_url: str) -> str:
    """
    Convert a gws-access LiCSAR frame URL to the data.ceda product root.

    From:
        https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/62/062D_05831_131313/

    To:
        https://data.ceda.ac.uk/neodc/comet/data/licsar_products/62/062D_05831_131313/
    """
    match = FRAME_RE.search(base_url)
    if not match:
        raise ValueError(
            "Could not parse track/frame from base URL. Expected something like "
            ".../LiCSAR_products/62/062D_05831_131313/"
        )

    track, frame = match.groups()
    return f"https://data.ceda.ac.uk/neodc/comet/data/licsar_products/{track}/{frame}/"


def filename_from_href_or_url(href: str) -> str:
    """
    Return only the filename from a relative href or absolute URL.
    """
    resolved = href.strip()
    parsed = urlparse(resolved)
    if parsed.scheme and parsed.netloc:
        return unquote(Path(parsed.path).name)
    return unquote(Path(resolved.strip("/")).name)


def is_absolute_url(href: str) -> bool:
    parsed = urlparse(href)
    return bool(parsed.scheme and parsed.netloc)


def resolve_file_url(
    href: str,
    listing_url: str,
    ceda_product_root: str,
    pair_name: str | None = None,
    metadata: bool = False,
) -> tuple[str, str]:
    """
    Resolve a listing href to a real download URL and local filename.

    Priority:
      1. If href is already absolute, use it.
      2. If this is an interferogram product, construct:
            ceda_product_root / pair_name / filename
      3. If this is metadata, construct:
            ceda_product_root / metadata / filename
      4. Fallback to urljoin(listing_url, href)
    """
    filename = filename_from_href_or_url(href)

    if is_absolute_url(href):
        return href, filename

    if pair_name is not None:
        return urljoin(ceda_product_root, f"{pair_name}/{filename}"), filename

    if metadata:
        return urljoin(ceda_product_root, f"metadata/{filename}"), filename

    return urljoin(listing_url.rstrip("/") + "/", href), filename


def find_ifg_pairs(
    session: requests.Session,
    base_url: str,
    cutoff: str,
    max_days: int | None,
    include_cutoff: bool,
) -> list[IfgPair]:
    ifg_listing_url = urljoin(base_url.rstrip("/") + "/", "interferograms/")
    cutoff_dt = parse_date(cutoff)

    hrefs = list_hrefs(session, ifg_listing_url)

    pairs: list[IfgPair] = []

    for href in hrefs:
        # Pair directories should be detected by their final path component.
        name = filename_from_href_or_url(href)
        match = PAIR_RE.match(name)
        if not match:
            continue

        d1, d2 = match.groups()
        start = parse_date(d1)
        end = parse_date(d2)

        if include_cutoff:
            if end > cutoff_dt:
                continue
        else:
            if end >= cutoff_dt:
                continue

        if max_days is not None and (end - start).days > max_days:
            continue

        # Use gws-access for pair listing discovery.
        pair_listing_url = urljoin(ifg_listing_url.rstrip("/") + "/", name)

        pairs.append(
            IfgPair(
                name=name,
                start=start,
                end=end,
                listing_url=pair_listing_url,
            )
        )

    pairs.sort(key=lambda p: (p.start, p.end))
    return pairs


def should_download_file(filename: str, products: set[str]) -> bool:
    """
    Product keywords:
        unw      -> *.geo.unw.tif
        cc       -> *.geo.cc.tif
        wrapped  -> *.geo.diff_pha.tif
        png      -> *.png
        all      -> everything in each interferogram directory
    """
    if "all" in products:
        return True

    if "unw" in products and filename.endswith(".geo.unw.tif"):
        return True

    if "cc" in products and filename.endswith(".geo.cc.tif"):
        return True

    if "wrapped" in products and filename.endswith(".geo.diff_pha.tif"):
        return True

    if "png" in products and filename.endswith(".png"):
        return True

    return False


def download_file(
    session: requests.Session,
    url: str,
    out_path: Path,
    overwrite: bool = False,
    retries: int = 3,
) -> tuple[Path, str, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        return out_path, "exists", url

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=180, allow_redirects=True) as response:
                response.raise_for_status()

                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            tmp_path.replace(out_path)
            return out_path, "downloaded", url

        except Exception as exc:
            last_error = exc

            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

            if attempt < retries:
                print(f"Retry {attempt}/{retries} after error for {url}")
                print(f"  {type(exc).__name__}: {exc}")
                time.sleep(2 * attempt)

    raise RuntimeError(f"Failed to download {url}: {last_error}") from last_error


def test_one_png(
    session: requests.Session,
    base_url: str,
    out_dir: Path,
    pair_name: str = "20161227_20170120",
) -> None:
    """
    Test one known PNG download using the expected CEDA product URL.

    This avoids ambiguity from directory parsing.
    """
    ceda_product_root = derive_ceda_product_root(base_url)

    filename = f"{pair_name}.geo.unw.png"
    test_url = urljoin(ceda_product_root, f"{pair_name}/{filename}")
    out_path = out_dir / "_test_download" / filename

    print("Testing one PNG download")
    print(f"URL: {test_url}")
    print(f"Output: {out_path}")

    path, status, used_url = download_file(
        session=session,
        url=test_url,
        out_path=out_path,
        overwrite=True,
        retries=2,
    )

    size = path.stat().st_size
    print(f"SUCCESS: {status} {path}")
    print(f"Size: {size:,} bytes")
    print(f"Used URL: {used_url}")


def download_metadata(
    session: requests.Session,
    base_url: str,
    ceda_product_root: str,
    out_dir: Path,
    overwrite: bool,
    workers: int,
) -> None:
    metadata_listing_url = urljoin(base_url.rstrip("/") + "/", "metadata/")
    metadata_out = out_dir / "metadata"

    keep_patterns = (
        ".geo.hgt.tif",
        ".geo.E.tif",
        ".geo.N.tif",
        ".geo.U.tif",
        ".geo.inc.tif",
        ".geo.landmask.tif",
        "metadata.txt",
        "data_summary.txt",
        "baselines",
        "network.png",
        "-poly.txt",
        ".azirg.csv",
    )

    try:
        hrefs = list_hrefs(session, metadata_listing_url)
    except Exception as exc:
        print("WARNING: Could not read metadata listing.")
        print(f"  Listing URL: {metadata_listing_url}")
        print(f"  {type(exc).__name__}: {exc}")
        print("Continuing without metadata.")
        return

    files: list[tuple[str, Path]] = []

    for href in hrefs:
        file_url, filename = resolve_file_url(
            href=href,
            listing_url=metadata_listing_url,
            ceda_product_root=ceda_product_root,
            metadata=True,
        )

        if not filename or filename == "..":
            continue

        if any(filename.endswith(p) or filename == p for p in keep_patterns):
            files.append((file_url, metadata_out / filename))

    # If the listing parser found nothing useful, try common expected files directly.
    if not files:
        frame_name = ceda_product_root.rstrip("/").split("/")[-1]
        expected = [
            f"{frame_name}.geo.hgt.tif",
            f"{frame_name}.geo.E.tif",
            f"{frame_name}.geo.N.tif",
            f"{frame_name}.geo.U.tif",
            "baselines",
            "network.png",
            "metadata.txt",
        ]

        for filename in expected:
            file_url = urljoin(ceda_product_root, f"metadata/{filename}")
            files.append((file_url, metadata_out / filename))

    print(f"Metadata files selected: {len(files)}")

    with futures.ThreadPoolExecutor(max_workers=workers) as ex:
        jobs = [
            ex.submit(download_file, session, url, path, overwrite)
            for url, path in files
        ]

        for job in futures.as_completed(jobs):
            try:
                path, status, used_url = job.result()
                print(f"  {status:10s} {path}")
            except Exception as exc:
                print(f"  WARNING metadata download failed: {type(exc).__name__}: {exc}")


def download_ifg_pair(
    session: requests.Session,
    pair: IfgPair,
    ceda_product_root: str,
    out_dir: Path,
    products: set[str],
    overwrite: bool,
) -> list[tuple[Path, str, str]]:
    hrefs = list_hrefs(session, pair.listing_url)

    selected: list[tuple[str, Path]] = []

    for href in hrefs:
        file_url, filename = resolve_file_url(
            href=href,
            listing_url=pair.listing_url,
            ceda_product_root=ceda_product_root,
            pair_name=pair.name,
        )

        if not filename or filename == "..":
            continue

        if should_download_file(filename, products):
            out_path = out_dir / "interferograms" / pair.name / filename
            selected.append((file_url, out_path))

    # If parser failed to recover file hrefs, construct expected filenames directly.
    if not selected:
        expected_filenames = []

        if "unw" in products or "all" in products:
            expected_filenames.append(f"{pair.name}.geo.unw.tif")
        if "cc" in products or "all" in products:
            expected_filenames.append(f"{pair.name}.geo.cc.tif")
        if "wrapped" in products or "all" in products:
            expected_filenames.append(f"{pair.name}.geo.diff_pha.tif")
        if "png" in products or "all" in products:
            expected_filenames.extend(
                [
                    f"{pair.name}.geo.unw.png",
                    f"{pair.name}.geo.cc.png",
                    f"{pair.name}.geo.diff.png",
                ]
            )

        for filename in expected_filenames:
            file_url = urljoin(ceda_product_root, f"{pair.name}/{filename}")
            out_path = out_dir / "interferograms" / pair.name / filename
            selected.append((file_url, out_path))

    results: list[tuple[Path, str, str]] = []

    for file_url, out_path in selected:
        print(f"Downloading: {file_url}")
        results.append(
            download_file(
                session=session,
                url=file_url,
                out_path=out_path,
                overwrite=overwrite,
            )
        )

    return results


def write_pair_list(pairs: Iterable[IfgPair], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "selected_ifg_pairs.csv"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("pair,start,end,temporal_baseline_days,listing_url\n")
        for p in pairs:
            f.write(
                f"{p.name},{p.start:%Y-%m-%d},{p.end:%Y-%m-%d},"
                f"{p.temporal_baseline_days},{p.listing_url}\n"
            )

    print(f"Wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-url",
        required=True,
        help="LiCSAR frame root URL ending in frame name.",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--cutoff",
        default="20170624",
        help="Cutoff date as YYYYMMDD. Default: 20170624.",
    )

    parser.add_argument(
        "--include-cutoff",
        action="store_true",
        help="Include interferograms ending on the cutoff date. Default excludes them.",
    )

    parser.add_argument(
        "--max-days",
        type=int,
        default=72,
        help="Maximum temporal baseline in days. Use --max-days 0 to disable.",
    )

    parser.add_argument(
        "--products",
        nargs="+",
        default=["unw", "cc"],
        choices=["unw", "cc", "wrapped", "png", "all"],
        help="Products to download from each interferogram directory.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel pair downloads.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List selected pairs but do not download.",
    )

    parser.add_argument(
        "--test-one-png",
        action="store_true",
        help="Download one known PNG and exit.",
    )

    parser.add_argument(
        "--test-pair",
        default="20161227_20170120",
        help="Pair name for --test-one-png. Default: 20161227_20170120.",
    )

    args = parser.parse_args()

    base_url = args.base_url.rstrip("/") + "/"
    out_dir = Path(args.out)
    products = set(args.products)
    max_days = None if args.max_days == 0 else args.max_days

    session = make_session()
    ceda_product_root = derive_ceda_product_root(base_url)

    print(f"Base listing URL: {base_url}")
    print(f"CEDA product root: {ceda_product_root}")
    print(f"Output directory: {out_dir}")

    if args.test_one_png:
        test_one_png(
            session=session,
            base_url=base_url,
            out_dir=out_dir,
            pair_name=args.test_pair,
        )
        return 0

    pairs = find_ifg_pairs(
        session=session,
        base_url=base_url,
        cutoff=args.cutoff,
        max_days=max_days,
        include_cutoff=args.include_cutoff,
    )

    print(f"\nFound {len(pairs)} interferogram pairs.")
    print(f"Cutoff: {'<=' if args.include_cutoff else '<'} {args.cutoff}")
    print(f"Max temporal baseline: {max_days if max_days is not None else 'disabled'} days")
    print(f"Products: {', '.join(sorted(products))}")

    if not pairs:
        print("No pairs matched. Try --max-days 0 or --include-cutoff.")
        return 1

    write_pair_list(pairs, out_dir)

    print("\nFirst 10 selected pairs:")
    for p in pairs[:10]:
        print(f"  {p.name}  dt={p.temporal_baseline_days:3d} days")

    print("\nLast 10 selected pairs:")
    for p in pairs[-10:]:
        print(f"  {p.name}  dt={p.temporal_baseline_days:3d} days")

    if args.dry_run:
        print("\nDry run only. No files downloaded.")
        return 0

    print("\nDownloading metadata...")
    download_metadata(
        session=session,
        base_url=base_url,
        ceda_product_root=ceda_product_root,
        out_dir=out_dir,
        overwrite=args.overwrite,
        workers=args.workers,
    )

    print("\nDownloading interferograms...")

    with futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        jobs = [
            ex.submit(
                download_ifg_pair,
                session,
                pair,
                ceda_product_root,
                out_dir,
                products,
                args.overwrite,
            )
            for pair in pairs
        ]

        completed = 0

        for job in futures.as_completed(jobs):
            completed += 1
            try:
                results = job.result()
                for path, status, used_url in results:
                    print(f"  {status:10s} {path}")
                print(f"Completed pair {completed}/{len(pairs)}")
            except Exception as exc:
                print(f"ERROR in pair job {completed}/{len(pairs)}")
                print(f"  {type(exc).__name__}: {exc}")

    print("\nDone.")
    print(f"Output directory: {out_dir.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())