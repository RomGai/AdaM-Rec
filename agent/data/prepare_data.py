"""Download, unpack and prepare Amazon metadata for all run_pipe datasets."""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd

if not __package__:
    # Support direct script execution as well as python -m agent.data.prepare_data.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.config.dataset_config import DatasetArgumentParser


META_SOURCES = {
    "beauty": ("meta_Beauty.json", "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz"),
    "clothing": ("meta_Clothing_Shoes_and_Jewelry.json", "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Clothing_Shoes_and_Jewelry.json.gz"),
    "music": ("meta_CDs_and_Vinyl.jsonl", "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_CDs_and_Vinyl.jsonl.gz"),
}


def _parse_legacy_meta_line(line: str) -> dict:
    text = line.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def _prepare_legacy_meta(raw_meta_path: Path, metadata_csv_path: Path, output_path: Path) -> dict:
    metadata_df = pd.read_csv(metadata_csv_path, dtype={"id": str})
    metadata_df["id"] = metadata_df["id"].astype(str).str.strip()
    metadata_df["price"] = pd.to_numeric(metadata_df["price"], errors="coerce")

    valid_ids = set(metadata_df["id"].tolist())
    price_map = {
        row["id"]: None if pd.isna(row["price"]) else float(row["price"])
        for _, row in metadata_df.iterrows()
    }

    kept = []
    total = 0
    dropped = 0
    assigned_price = 0

    with raw_meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            rec = _parse_legacy_meta_line(line)
            asin = str(rec.get("asin", "")).strip()
            if not asin or asin not in valid_ids:
                dropped += 1
                continue
            if rec.get("price") in (None, "", "NaN"):
                p = price_map.get(asin)
                if p is not None:
                    rec["price"] = p
                    assigned_price += 1
            kept.append(rec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "raw_total": total,
        "kept": len(kept),
        "dropped": dropped,
        "assigned_price": assigned_price,
        "output": str(output_path),
    }


def _parse_2023_meta_line(line: str) -> dict[str, Any]:
    text = line.strip()
    if not text:
        return {}
    return json.loads(text)


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict, tuple, set)) and not value:
            continue
        return value
    return None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_normalize_text(x) for x in value]
        return " ".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            text = _normalize_text(v)
            if text:
                parts.append(f"{k}: {text}")
        return "; ".join(parts).strip()
    return str(value).strip()


def _normalize_price(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().replace("$", "").replace(",", "")
        if not cleaned:
            return None
        value = cleaned
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return None
    return float(parsed)


def _normalize_categories(raw_categories: Any, fallback_category: Any = None) -> list[list[str]]:
    source = raw_categories
    if source is None or source == "":
        source = fallback_category

    if isinstance(source, list):
        if not source:
            return []
        if all(isinstance(x, str) for x in source):
            segs = [x.strip() for x in source if str(x).strip()]
            return [segs] if segs else []

        out: list[list[str]] = []
        for path in source:
            if isinstance(path, list):
                segs = [str(x).strip() for x in path if str(x).strip()]
                if segs:
                    out.append(segs)
            elif isinstance(path, str) and path.strip():
                segs = [seg.strip() for seg in path.split("|") if seg.strip()]
                if segs:
                    out.append(segs)
        return out

    if isinstance(source, str) and source.strip():
        normalized = source.replace(">", "|")
        segs = [seg.strip() for seg in normalized.split("|") if seg.strip()]
        return [segs] if segs else []

    return []


def _extract_image_url(record: dict[str, Any]) -> str:
    images = record.get("images")
    if isinstance(images, list):
        for image in images:
            if isinstance(image, dict):
                url = _first_non_empty(image.get("hi_res"), image.get("large"), image.get("thumb"), image.get("url"))
                if isinstance(url, str) and url.strip():
                    return url.strip()
            elif isinstance(image, str) and image.strip():
                return image.strip()

    direct = _first_non_empty(record.get("imUrl"), record.get("image"), record.get("image_url"))
    return direct.strip() if isinstance(direct, str) else ""


def _extract_related(record: dict[str, Any]) -> dict[str, Any]:
    related = record.get("related")
    if isinstance(related, dict):
        return {k: v for k, v in related.items() if v not in (None, [], {}, "")}

    out: dict[str, Any] = {}
    for key in ("also_bought", "also_viewed", "bought_together", "buy_after_viewing"):
        value = record.get(key)
        if value not in (None, [], {}, ""):
            out[key] = value
    return out


def _normalize_sales_rank(record: dict[str, Any], metadata_row: dict[str, Any] | None) -> dict[str, Any]:
    sales_rank = record.get("salesRank")
    if isinstance(sales_rank, dict) and sales_rank:
        return sales_rank

    if metadata_row:
        ranking = metadata_row.get("ranking")
        category = _normalize_text(_first_non_empty(record.get("main_category"), metadata_row.get("category")))
        if pd.notna(ranking):
            try:
                ranking_value = int(float(ranking))
            except (TypeError, ValueError):
                ranking_value = None
            if ranking_value is not None and category:
                return {category: ranking_value}
    return {}


def _load_metadata(metadata_csv_path: Path) -> tuple[set[str], dict[str, dict[str, Any]], dict[str, float | None]]:
    metadata_df = pd.read_csv(metadata_csv_path, dtype={"id": str})
    metadata_df["id"] = metadata_df["id"].astype(str).str.strip()
    if "price" in metadata_df.columns:
        metadata_df["price"] = pd.to_numeric(metadata_df["price"], errors="coerce")

    valid_ids = set(metadata_df["id"].tolist())
    metadata_rows = {
        row["id"]: row.to_dict()
        for _, row in metadata_df.iterrows()
    }
    price_map = {
        row["id"]: None if "price" not in row or pd.isna(row["price"]) else float(row["price"])
        for _, row in metadata_df.iterrows()
    }
    return valid_ids, metadata_rows, price_map


def _canonicalize_record(record: dict[str, Any], item_id: str, metadata_row: dict[str, Any] | None, fallback_price: float | None) -> tuple[dict[str, Any], bool]:
    normalized = dict(record)

    normalized["asin"] = item_id
    if "parent_asin" not in normalized:
        normalized["parent_asin"] = item_id

    title = _normalize_text(_first_non_empty(record.get("title"), metadata_row.get("title") if metadata_row else None))
    if title:
        normalized["title"] = title

    description = _normalize_text(_first_non_empty(record.get("description"), metadata_row.get("description") if metadata_row else None, record.get("features")))
    normalized["description"] = description

    categories = _normalize_categories(record.get("categories"), metadata_row.get("category") if metadata_row else None)
    normalized["categories"] = categories

    image_url = _extract_image_url(record)
    if image_url:
        normalized["imUrl"] = image_url

    related = _extract_related(record)
    if related:
        normalized["related"] = related

    price = _normalize_price(record.get("price"))
    assigned_price = False
    if price is None and fallback_price is not None:
        price = fallback_price
        assigned_price = True
    if price is not None:
        normalized["price"] = price

    sales_rank = _normalize_sales_rank(record, metadata_row)
    if sales_rank:
        normalized["salesRank"] = sales_rank

    return normalized, assigned_price


def _prepare_2023_meta(raw_meta_path: Path, metadata_csv_path: Path, output_path: Path) -> dict[str, Any]:
    valid_ids, metadata_rows, price_map = _load_metadata(metadata_csv_path)

    kept: list[dict[str, Any]] = []
    total = 0
    dropped = 0
    assigned_price = 0

    with raw_meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            record = _parse_2023_meta_line(line)
            item_id = _normalize_text(
                _first_non_empty(
                    record.get("asin"),
                    record.get("parent_asin"),
                    record.get("id"),
                    record.get("item_id"),
                )
            )
            if not item_id or item_id not in valid_ids:
                dropped += 1
                continue

            normalized, price_was_assigned = _canonicalize_record(
                record=record,
                item_id=item_id,
                metadata_row=metadata_rows.get(item_id),
                fallback_price=price_map.get(item_id),
            )
            if price_was_assigned:
                assigned_price += 1
            kept.append(normalized)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in kept:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "raw_total": total,
        "kept": len(kept),
        "dropped": dropped,
        "assigned_price": assigned_price,
        "output": str(output_path),
    }


def _ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _temporary_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=target.name + ".", suffix=".part", dir=target.parent)
    os.close(fd)
    return Path(name)


def _copy_stream(source, destination, label: str) -> int:
    copied = 0
    last_report = time.monotonic()
    while chunk := source.read(1024 * 1024):
        destination.write(chunk)
        copied += len(chunk)
        if time.monotonic() - last_report >= 10:
            print(f"[Data] {label}: {copied / (1024 ** 2):.1f} MiB", flush=True)
            last_report = time.monotonic()
    return copied


def _download_gzip(url: str, target: Path) -> None:
    temporary = _temporary_path(target)
    try:
        print(f"[Data] download {url} -> {target}", flush=True)
        request = Request(url, headers={"User-Agent": "AdaM-Rec/1.0", "Accept-Encoding": "identity"})
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            copied = _copy_stream(response, output, "download")
            length = response.headers.get("Content-Length")
            if length is not None and copied != int(length):
                raise OSError(f"Incomplete download: expected {length} bytes, received {copied}")
        with temporary.open("rb") as downloaded:
            if downloaded.read(2) != b"\x1f\x8b":
                raise ValueError(f"Download is not a gzip archive: {url}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _unpack_gzip(archive: Path, target: Path) -> None:
    temporary = _temporary_path(target)
    try:
        print(f"[Data] decompress {archive} -> {target}", flush=True)
        with gzip.open(archive, "rb") as source, temporary.open("wb") as output:
            copied = _copy_stream(source, output, "decompress")
        if copied == 0:
            raise ValueError(f"Empty metadata archive: {archive}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _check_csv(path: Path, required: set[str], query: bool = False) -> None:
    if not _ready(path):
        raise FileNotFoundError(f"Required local CSV is missing or empty: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        fields = set(next(csv.reader(source), []))
    if required - fields:
        raise ValueError(f"{path}: missing columns {sorted(required - fields)}")
    if query and not ({"query", "new_query"} & fields):
        raise ValueError(f"{path}: requires query or new_query column")


def ensure_dataset_data(args, *, require_query: bool = True) -> dict:
    """Reuse local data or prepare the selected dataset using its official source."""
    query = Path(args.query_csv)
    filtered = Path(args.filtered_meta_jsonl)
    if require_query:
        _check_csv(query, {"id", "user_id"}, query=True)
    if _ready(filtered):
        print(f"[Data] reuse filtered metadata: {filtered}")
        return {"status": "reused", "output": str(filtered)}
    if not getattr(args, "auto_prepare_data", True):
        raise FileNotFoundError(f"Filtered metadata is missing or empty: {filtered}; enable --auto-prepare-data or prepare it manually")

    metadata = Path(getattr(args, "metadata_csv", None) or query.parent / "metadata.csv")
    required = {"id"} if args.dataset == "music" else {"id", "price"}
    _check_csv(metadata, required)
    filename, url = META_SOURCES[args.dataset]
    explicit_raw = getattr(args, "raw_meta", None)
    if explicit_raw:
        supplied = Path(explicit_raw)
        raw = supplied.with_suffix("") if supplied.suffix.lower() == ".gz" else supplied
        archive = supplied if supplied.suffix.lower() == ".gz" else Path(str(raw) + ".gz")
    else:
        # Also recognize files downloaded with wget in the working directory.
        raw_candidates = list(dict.fromkeys([filtered.parent / filename, metadata.parent / filename, Path(filename)]))
        raw = next((p for p in raw_candidates if _ready(p)), raw_candidates[0])
        archive = next((Path(str(p) + ".gz") for p in raw_candidates if _ready(Path(str(p) + ".gz"))), Path(str(raw) + ".gz"))

    if filtered.resolve() in {raw.resolve(), archive.resolve(), metadata.resolve(), query.resolve()}:
        raise ValueError("Filtered output must not overwrite an input file")
    if not _ready(raw):
        if not _ready(archive):
            if not getattr(args, "download_missing_data", True):
                raise FileNotFoundError(f"No local raw metadata or gzip archive for {args.dataset}; downloads disabled")
            _download_gzip(url, archive)
        _unpack_gzip(archive, raw)
    else:
        print(f"[Data] reuse raw metadata: {raw}")

    temporary = _temporary_path(filtered)
    try:
        print(f"[Data] preprocess {raw} using {metadata}", flush=True)
        prepare = _prepare_2023_meta if args.dataset == "music" else _prepare_legacy_meta
        summary = prepare(raw, metadata, temporary)
        if int(summary.get("kept", 0)) <= 0 or not _ready(temporary):
            raise ValueError("Preprocessing matched no items; check dataset selection and metadata.csv IDs")
        temporary.replace(filtered)
    finally:
        temporary.unlink(missing_ok=True)
    summary.update(status="prepared", output=str(filtered), raw_meta=str(raw), metadata_csv=str(metadata))
    print(f"[Data] ready: {json.dumps(summary, ensure_ascii=False)}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = DatasetArgumentParser(description="Prepare Beauty, Clothing or Music metadata; reuse local files before downloading")
    parser.add_argument("--dataset", choices=["auto", "beauty", "clothing", "music"], default="auto")
    parser.add_argument("--raw-meta", default=None, help="Local raw metadata or gzip archive")
    parser.add_argument("--metadata-csv", default=None, help="Item ID filter CSV; defaults to the dataset query directory")
    parser.add_argument("--output", "--filtered-meta-jsonl", dest="filtered_meta_jsonl", default=None, help="Filtered JSONL output; defaults to the selected dataset")
    parser.add_argument("--query-csv", default=None, help="Optional path used for dataset inference and locating metadata.csv; query file is not required by this standalone tool")
    parser.add_argument("--download-missing-data", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    summary = ensure_dataset_data(args, require_query=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
