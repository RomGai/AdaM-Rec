"""Text/VL profiling prompts, concurrent generation and SQLite caches for run_pipe."""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from agent.profiling.vllm_item_profiler import VLLMItemProfiler


BehaviorLabel = Literal["positive", "negative"]


@dataclass
class ItemProfileInput:
    item_id: str
    title: str
    detail_text: str
    main_image: str
    detail_images: List[str] = field(default_factory=list)
    price: Optional[str] = None
    brand: Optional[str] = None
    category_hint: Optional[str] = None


@dataclass
class HistoryItemProfileInput(ItemProfileInput):
    user_id: str = ""
    behavior: BehaviorLabel = "positive"
    timestamp: Optional[int] = 0


def _normalize_timestamp_for_db(ts: Optional[int]) -> int:
    """Normalize optional timestamp to DB-safe integer.

    Negative samples may have no real interaction timestamp. We store such cases as -1.
    """
    if ts is None:
        return -1
    return int(ts)


class GlobalItemDB:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS global_item_features (
                item_id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def upsert(self, item_id: str, profile: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO global_item_features (item_id, profile_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(item_id) DO UPDATE SET
                profile_json=excluded.profile_json,
                updated_at=excluded.updated_at
            """,
            (
                item_id,
                json.dumps(profile, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    def get_profile(self, item_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT profile_json FROM global_item_features WHERE item_id = ?",
            (str(item_id),),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0])


class UserHistoryLogDB:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_history_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                behavior TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                profile_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_user_time
            ON user_history_profiles (user_id, timestamp)
            """
        )
        self.conn.commit()

    def insert(
        self,
        user_id: str,
        item_id: str,
        behavior: BehaviorLabel,
        timestamp: Optional[int],
        profile: Dict[str, Any],
    ) -> None:
        timestamp_db = _normalize_timestamp_for_db(timestamp)
        self.conn.execute(
            """
            INSERT INTO user_history_profiles
            (user_id, item_id, behavior, timestamp, profile_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                item_id,
                behavior,
                timestamp_db,
                json.dumps(profile, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    def exists(
        self,
        user_id: str,
        item_id: str,
        behavior: BehaviorLabel,
        timestamp: Optional[int],
    ) -> bool:
        timestamp_db = _normalize_timestamp_for_db(timestamp)
        cursor = self.conn.execute(
            """
            SELECT 1 FROM user_history_profiles
            WHERE user_id = ? AND item_id = ? AND behavior = ? AND timestamp = ?
            LIMIT 1
            """,
            (str(user_id), str(item_id), str(behavior), timestamp_db),
        )
        return cursor.fetchone() is not None


def build_vl_profile_prompt(item: ItemProfileInput) -> str:
    """Prompt template for fine-grained textual + visual profiling."""

    image_count = 1 + len(item.detail_images)
    return f"""
You are an expert e-commerce item profiler.
Given product text and {image_count} images (first is main image, rest are detail images),
extract a fine-grained feature profile in STRICT JSON.

Item text fields:
- title: {item.title}
- detail_text: {item.detail_text}
- brand: {item.brand or ''}
- price: {item.price or ''}
- category_hint: {item.category_hint or ''}

User shopping-oriented extraction requirements:
1) Type-first taxonomy (important):
   - Always output `item_type` (required).
   - If hierarchical category is uncertain, keep only `item_type` and leave `category_path` empty.
   - If known, output `category_path` as a list (e.g., ["Electronics", "Gaming", "Headset"]).
   - Also infer use_case, target_people, seasonality.
2) Textual attribute tags (fine-grained):
   - title keyword summary (must leverage title)
   - material/fabric composition
   - core features & specs (size, capacity, weight, dimensions, compatibility, power, ingredients)
   - package/bundle information
   - quality & durability claims
   - comfort/usability claims
   - price_band inference (budget/mid/premium) and value_for_money signal
3) Visual style tags (from images):
   - dominant colors (+ optional hex-like names)
   - silhouette/shape/form factor
   - style keywords (minimalist, sporty, retro, luxury, kawaii, etc.)
   - texture/finish (matte/glossy/metallic/knit/grainy)
   - pattern/print/logo density
   - scene mood (homey, professional, outdoor, gaming, etc.)
   - perceived quality level (low/medium/high with confidence)
4) Output quality:
   - Every major field must include a confidence in [0,1].
   - Put uncertain values under "hypotheses".
   - Output ONLY one JSON object. No markdown.

JSON schema:
{{
  "item_id": "{item.item_id}",
  "title": "{item.title}",
  "taxonomy": {{
    "item_type": "",
    "category_path": [],
    "use_case": [],
    "target_people": [],
    "seasonality": "",
    "confidence": 0.0
  }},
  "text_tags": {{...}},
  "visual_tags": {{...}},
  "hypotheses": ["..."],
  "overall_confidence": 0.0
}}
""".strip()


def build_text_profile_prompt(item: ItemProfileInput) -> str:
    """Text-only counterpart of the original English profiling prompt."""
    return f"""
You are an expert e-commerce item profiler.
Given product text only, extract a fine-grained feature profile in STRICT JSON.
No images are provided.

Item text fields:
- title: {item.title}
- detail_text: {item.detail_text}
- brand: {item.brand or ''}
- price: {item.price or ''}
- category_hint: {item.category_hint or ''}

User shopping-oriented extraction requirements:
1) Type-first taxonomy (important):
   - Always output `item_type` (required).
   - If hierarchical category is uncertain, keep only `item_type` and leave `category_path` empty.
   - If known, output `category_path` as a list (e.g., ["Electronics", "Gaming", "Headset"]).
   - Also infer use_case, target_people, seasonality.
2) Textual attribute tags (fine-grained):
   - title keyword summary (must leverage title)
   - material/fabric composition
   - core features & specs (size, capacity, weight, dimensions, compatibility, power, ingredients)
   - package/bundle information
   - quality & durability claims
   - comfort/usability claims
   - price_band inference (budget/mid/premium) and value_for_money signal
3) Visual style information in text:
   - No images are provided; do not claim to observe visual features from images.
   - If the product text explicitly describes colors, silhouette/shape/form factor,
     style, texture/finish, pattern/print/logo density, or scene mood,
     include these descriptions in `text_tags`.
   - Keep `visual_tags` as an empty object.
4) Output quality:
   - Every major populated field must include a confidence in [0,1].
   - Put uncertain values under "hypotheses".
   - Output ONLY one JSON object. No markdown.

JSON schema:
{{
  "item_id": "{item.item_id}",
  "title": "{item.title}",
  "taxonomy": {{
    "item_type": "",
    "category_path": [],
    "use_case": [],
    "target_people": [],
    "seasonality": "",
    "confidence": 0.0
  }},
  "text_tags": {{...}},
  "visual_tags": {{}},
  "hypotheses": ["..."],
  "overall_confidence": 0.0
}}
""".strip()


def build_profile_prompt(item: ItemProfileInput, use_vl: bool = True) -> str:
    """Keep existing callers on the original VL prompt by default."""
    return build_vl_profile_prompt(item) if use_vl else build_text_profile_prompt(item)


def get_or_create_item_profiles(
    extractor: VLLMItemProfiler,
    global_db: GlobalItemDB,
    items: List[ItemProfileInput],
    use_vl: bool,
    concurrency: int = 1,
    progress=None,
) -> List[Dict[str, Any]]:
    """Generate cache misses concurrently; keep SQLite writes on the caller thread."""
    if concurrency < 1:
        raise ValueError("profile concurrency must be positive")
    generation = {
        "prompt_version": "english_split_v1",
        "mode": "vl" if use_vl else "text",
        "model": extractor.model_name,
    }
    profiles = {}
    missing = {}
    for item in items:
        if item.item_id in profiles or item.item_id in missing:
            continue
        cached = global_db.get_profile(item.item_id)
        if cached is not None and cached.get("_profile_generation") == generation:
            profiles[item.item_id] = cached
        else:
            missing[item.item_id] = item
    cached_count = len(profiles)
    total = cached_count + len(missing)
    if progress:
        progress(len(profiles), total, cached_count)

    def generate(item):
        profile = extractor.extract(
            prompt=build_profile_prompt(item, use_vl=use_vl),
            image_paths=[item.main_image, *item.detail_images] if use_vl else [],
            text_only_prompt=build_text_profile_prompt(item),
        )
        if not use_vl:
            profile["visual_tags"] = {}
        profile["_profile_generation"] = generation
        return profile

    def save(item, profile):
        global_db.upsert(item.item_id, profile)
        profiles[item.item_id] = profile
        if progress:
            progress(len(profiles), total, cached_count)

    if concurrency == 1:
        for item in missing.values():
            save(item, generate(item))
    elif missing:
        remaining = iter(missing.values())
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            pending = {}

            def submit_next():
                item = next(remaining, None)
                if item is not None:
                    pending[executor.submit(generate, item)] = item

            for _ in range(min(concurrency, len(missing))):
                submit_next()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                failures = []
                for future in done:
                    item = pending.pop(future)
                    try:
                        save(item, future.result())
                    except Exception as exc:
                        failures.append(exc)
                if failures:
                    # Preserve already-running successes before reporting the failure.
                    for future, item in pending.items():
                        try:
                            save(item, future.result())
                        except Exception:
                            pass
                    raise failures[0]
                for _ in done:
                    submit_next()
    return [profiles[item.item_id] for item in items]
