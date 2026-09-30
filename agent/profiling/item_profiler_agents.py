"""Text/VL item profiling prompts, generation and SQLite caches for run_pipe."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from agent.models.qwen35_backbone import load_backbone


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


class QwenItemProfiler:
    """Generate text or text/image profiles with the shared Qwen3.5 backbone."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-9B",
        device: str = "cuda",
        torch_dtype: str = "auto",
        max_new_tokens: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens or int(os.getenv("out_seq_length", "16384"))
        self.device = device
        self.torch_dtype = torch_dtype
        self.do_sample = os.getenv("greedy", "false").lower() != "true"
        self.top_p = float(os.getenv("top_p", "0.8"))
        self.top_k = int(os.getenv("top_k", "20"))
        self.temperature = float(os.getenv("temperature", "0.7"))
        self.repetition_penalty = float(os.getenv("repetition_penalty", "1.0"))
        self.json_retry = int(os.getenv("json_retry", "1"))
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is None:
            self._model, self._processor = load_backbone(
                self.model_name, self.torch_dtype,
                "auto" if self.device == "cuda" else self.device,
            )

    def _generate_text(self, messages: List[Dict[str, Any]], force_greedy: bool = False) -> str:
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            enable_thinking=False,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False if force_greedy else self.do_sample,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "temperature": 0.0 if force_greedy else self.temperature,
            "repetition_penalty": self.repetition_penalty,
        }
        if force_greedy:
            generate_kwargs.pop("top_p", None)
            generate_kwargs.pop("top_k", None)

        output_ids = self._model.generate(**inputs, **generate_kwargs)

        generated_ids = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        generated_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return generated_text

    @staticmethod
    def _try_json_decode(text: str) -> Optional[Dict[str, Any]]:
        decoder = json.JSONDecoder()
        stripped = text.strip()

        # 1) direct decode
        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        # 2) markdown json code fence
        if "```" in stripped:
            parts = stripped.split("```")
            for part in parts:
                candidate = part.replace("json", "", 1).strip()
                if not candidate:
                    continue
                try:
                    payload = json.loads(candidate)
                    if isinstance(payload, dict):
                        return payload
                except json.JSONDecodeError:
                    continue

        # 3) find first decodable JSON object from any '{' start
        for i, ch in enumerate(stripped):
            if ch != "{":
                continue
            try:
                payload, _end = decoder.raw_decode(stripped, i)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload

        return None



    @staticmethod
    def _normalize_image_paths(image_paths: List[str]) -> List[str]:
        """Drop empty/obviously invalid image entries before processor ingestion."""
        cleaned: List[str] = []
        for p in image_paths:
            cand = str(p or "").strip()
            if not cand:
                continue
            # Avoid placeholders that are known to break image loading.
            if cand in {".", "./", "..", "../"}:
                continue
            cleaned.append(cand)
        return cleaned

    def extract(
        self,
        prompt: str,
        image_paths: List[str],
        text_only_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.load()

        valid_image_paths = self._normalize_image_paths(image_paths)
        image_messages = [{"type": "image", "image": path} for path in valid_image_paths]
        if not image_messages and text_only_prompt is not None:
            prompt = text_only_prompt
        messages = [
            {
                "role": "user",
                "content": [
                    *image_messages,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            generated_text = self._generate_text(messages)
        except Exception as exc:
            if not image_messages:
                raise
            # Some rows contain invalid image URLs/paths; fallback to text-only profiling.
            print(f"[QwenItemProfiler] image loading failed, fallback to text-only: {exc}")
            image_messages = []
            if text_only_prompt is not None:
                prompt = text_only_prompt
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            generated_text = self._generate_text(messages)

        parsed = self._try_json_decode(generated_text)
        if parsed is not None:
            if not image_messages and text_only_prompt is not None:
                parsed["visual_tags"] = {}
            return parsed

        # Retry with stricter formatting instruction to reduce JSON parsing failures.
        for retry_idx in range(self.json_retry):
            strict_messages = [
                {
                    "role": "user",
                    "content": [
                        *image_messages,
                        {
                            "type": "text",
                            "text": (
                                prompt
                                + "\n\nIMPORTANT: Output exactly one valid JSON object only. "
                                + "Do not include markdown/code fences/comments/trailing text."
                            ),
                        },
                    ],
                }
            ]
            if not image_messages:
                strict_messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    prompt
                                    + "\n\nIMPORTANT: Output exactly one valid JSON object only. "
                                    + "Do not include markdown/code fences/comments/trailing text."
                                ),
                            }
                        ],
                    }
                ]
            generated_text = self._generate_text(strict_messages, force_greedy=True)
            parsed = self._try_json_decode(generated_text)
            if parsed is not None:
                if not image_messages and text_only_prompt is not None:
                    parsed["visual_tags"] = {}
                return parsed

        raise ValueError(
            "Model output is not valid JSON after retries. "
            f"Last output (truncated): {generated_text[:2000]}"
        )



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


def get_or_create_item_profile(
    extractor: QwenItemProfiler,
    global_db: GlobalItemDB,
    item: ItemProfileInput,
    use_vl: bool,
) -> Dict[str, Any]:
    """Reuse only profiles generated with the same prompt version, mode and backbone."""
    generation = {
        "prompt_version": "english_split_v1",
        "mode": "vl" if use_vl else "text",
        "model": extractor.model_name,
    }
    cached = global_db.get_profile(item.item_id)
    if cached is not None and cached.get("_profile_generation") == generation:
        return cached

    image_paths = [item.main_image, *item.detail_images] if use_vl else []
    profile = extractor.extract(
        prompt=build_profile_prompt(item, use_vl=use_vl),
        image_paths=image_paths,
        text_only_prompt=build_text_profile_prompt(item),
    )
    if not use_vl:
        profile["visual_tags"] = {}
    profile["_profile_generation"] = generation
    global_db.upsert(item.item_id, profile)
    return profile
