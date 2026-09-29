"""LLM routing, pseudo-query rewriting and recall-budget control for run_pipe."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from agent.models.qwen35_backbone import load_text_backbone


@dataclass
class RoutingResult:
    query: str
    category_paths: List[List[str]]
    item_types: List[str]
    reasoning: str


class QwenRouterLLM:
    """Use the shared Qwen3.5 backbone for routing and recall control."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-9B",
        max_new_tokens: int = 2048,
        enable_thinking: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        if self._model is None:
            self._model, self._tokenizer = load_text_backbone(self.model_name)

    @staticmethod
    def _try_json_decode(text: str) -> Optional[Dict[str, Any]]:
        stripped = text.strip()
        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        if "```" in stripped:
            for part in stripped.split("```"):
                cand = part.replace("json", "", 1).strip()
                if not cand:
                    continue
                try:
                    payload = json.loads(cand)
                    if isinstance(payload, dict):
                        return payload
                except json.JSONDecodeError:
                    continue
        return None

    def route(
        self,
        query: str,
        category_catalog: Sequence[str],
        item_type_catalog: Sequence[str],
    ) -> RoutingResult:
        self.load()

        catalog_text = "\n".join(f"- {c}" for c in category_catalog[:300])
        item_type_text = "\n".join(f"- {i}" for i in item_type_catalog[:300])
        prompt = (
            "你是电商检索路由专家。请把用户Query映射到给定类目/类型；若都不匹配，可新造一个合理类目。\n"
            "输出必须是一个JSON对象，字段:"
            "category_paths(二维数组，每条是层级路径), item_types(数组), reasoning(字符串)。\n\n"
            f"用户Query: {query}\n\n"
            "候选类目路径清单:\n"
            f"{catalog_text}\n\n"
            "候选item_type清单:\n"
            f"{item_type_text}\n"
        )

        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        generated_ids = self._model.generate(**model_inputs, max_new_tokens=self.max_new_tokens)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        # Read the answer after the optional thinking section.
        try:
            index = len(output_ids) - output_ids[::-1].index(self._tokenizer.convert_tokens_to_ids("</think>"))
        except ValueError:
            index = 0
        content = self._tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        payload = self._try_json_decode(content)
        if payload is None:
            payload = {
                "category_paths": [],
                "item_types": [],
                "reasoning": f"Failed to parse JSON from LLM output: {content[:500]}",
            }

        raw_paths = payload.get("category_paths", [])
        category_paths: List[List[str]] = []
        for p in raw_paths:
            if isinstance(p, list):
                segs = [str(x).strip() for x in p if str(x).strip()]
            else:
                segs = [x.strip() for x in str(p).replace("/", ">").split(">") if x.strip()]
            if segs:
                category_paths.append(segs)

        item_types = [str(x).strip() for x in payload.get("item_types", []) if str(x).strip()]
        return RoutingResult(
            query=query,
            category_paths=category_paths,
            item_types=item_types,
            reasoning=str(payload.get("reasoning", "")),
        )

    def rewrite_pseudo_query(self, real_query: str, history_item_info: Dict[str, Any]) -> str:
        """Generate a history-grounded pseudo query using the real query as a style/reference example."""
        self.load()
        clean_query = str(real_query or "").strip()
        if not clean_query:
            return ""

        history_item_json = json.dumps(history_item_info or {}, ensure_ascii=False)
        prompt = (
            "Task:\n"
            "Use the example query as a reference style, then write a NEW query for this history item.\n"
            "The output query should belong to the history item, not a rewrite/copy of the example query.\n\n"
            "Guidance:\n"
            "1) Treat the example query as style and granularity reference (how concise, what kind of slots), not fixed content.\n"
            "2) If a slot from the example query has no evidence in the history item, you may omit or soften it.\n"
            "3) Avoid fabrication; only use information supported by the history item.\n\n"
            "Input:\n"
            f"Example query:\n{clean_query}\n\n"
            f"History item info (JSON):\n{history_item_json}\n\n"
            "Output:\n"
            "<one-line new query for the history item>"
        )
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=min(256, int(self.max_new_tokens)),
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        try:
            index = len(output_ids) - output_ids[::-1].index(self._tokenizer.convert_tokens_to_ids("</think>"))
        except ValueError:
            index = 0
        content = self._tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n").strip()
        if not content:
            return ""

        # Robustly strip leaked thinking tags/chunks and pick a meaningful query line.
        cleaned = re.sub(r"<\s*think\s*>.*?<\s*/\s*think\s*>", " ", content, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"<\s*/?\s*think\s*>", " ", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace("</think>", " ").replace("<think>", " ")

        candidate_lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        kept_lines: List[str] = []
        for line in candidate_lines:
            lowered = line.lower()
            if lowered in {"<think>", "</think>", "think", "thinking"}:
                continue
            normalized = re.sub(r"\s+", " ", line).strip("\"'` ")
            if normalized:
                kept_lines.append(normalized)
        if not kept_lines:
            return ""
        merged = " ".join(kept_lines).strip()
        payload = self._try_json_decode(merged)
        if isinstance(payload, dict):
            cand = str(payload.get("pseudo_query", "") or "").strip()
            if cand:
                merged = cand
        return merged

    def optimize_modulation_params(
        self,
        query: str,
        step: int,
        target_item_id: str,
        pseudo_query: str,
        text_rank: int,
        vl_rank: int,
        current_text_weight: float,
        current_vl_weight: float,
        current_total_recall: int,
        rule_hints: Dict[str, Any],
        recent_memory: Sequence[Dict[str, Any]],
        min_total_recall: int,
        max_total_recall: int,
    ) -> Optional[Dict[str, Any]]:
        """Propose weights and budget; run_pipe currently consumes the budget only."""
        recall_floor = max(1, int(min_total_recall))
        recall_ceiling = max(recall_floor, int(max_total_recall))
        self.load()
        memory_window = list(recent_memory[-5:]) if recent_memory else []
        context_payload = {
            "step": int(step),
            "target_item_id": str(target_item_id or ""),
            "pseudo_query": str(pseudo_query or ""),
            "observation": {"text_rank": int(text_rank), "vl_rank": int(vl_rank)},
            "current_state": {
                "text_weight": float(current_text_weight),
                "vl_weight": float(current_vl_weight),
                "total_recall": int(current_total_recall),
            },
            "rule_hints": rule_hints,
            "recent_memory": memory_window,
            "constraints": {
                "text_weight_range": [0.05, 0.95],
                "vl_weight_sum_rule": "vl_weight = 1 - text_weight",
                "prefer_large_divergence_band": [0.2, 0.8],
                "total_recall_range": [recall_floor, recall_ceiling],
                "max_step_weight_change": 0.18,
                "output_keys": ["text_weight", "vl_weight", "total_recall", "summary", "reasoning"],
            },
        }
        prompt = (
            "You are Agent3, a modality routing optimizer for AdaM-Rec.\n"
            "Goal: update text_weight, vl_weight, and total_recall for this iteration using the provided observations.\n\n"
            "Core objective:\n"
            "- Dynamically decide how much to rely on text-based recall vs. vision-language recall.\n"
            "- Maximize useful differentiation between text and vl weights when confidence is high.\n"
            "- Avoid near-equal weights unless evidence is genuinely mixed or uncertain.\n\n"
            "Important routing principle:\n"
            "The preferred differentiation band is [0.2, 0.8]. "
            "When justified, push the dominant modality near >=0.8 and the weaker modality near <=0.2. "
            "Use moderate weights such as 0.65/0.35 or 0.7/0.3 when evidence is suggestive but not decisive. "
            "Use balanced weights only when both modalities are similarly useful or evidence is weak.\n\n"
            "Text-dominant guidance:\n"
            "Increase text_weight when the query or pseudo-query evidence is mainly semantic, functional, or specification-driven. "
            "Typical text-driven signals include brand, model, compatibility, product type, function, material, size, capacity, wattage, voltage, flavor, scent, ingredient, quantity, title, version, or other explicit constraints. "
            "If text recall ranks the target clearly better than VL recall, or VL retrieves visually similar but functionally wrong items, strongly favor text. "
            "When these signals are strong and consistent, set text_weight >= 0.8.\n\n"
            "VL-dominant guidance:\n"
            "Increase vl_weight when the query or pseudo-query evidence depends mainly on visual appearance. "
            "Typical VL-driven signals include style, color, shape, silhouette, texture, pattern, decoration, aesthetic, design, look, packaging, or visual similarity. "
            "If VL recall ranks the target clearly better than text recall, strongly favor VL. "
            "When these signals are strong and consistent, set vl_weight >= 0.8.\n\n"
            "Mixed routing guidance:\n"
            "Use moderate or balanced weights when the query combines textual constraints with visual preferences, "
            "or when text and VL ranks are close, alternate wins, or provide complementary evidence. "
            "Prefer mild differentiation such as 0.6/0.4 or 0.7/0.3 over exactly 0.5/0.5 when there is any clear tendency.\n\n"
            "Observation interpretation:\n"
            "- Lower rank is better.\n"
            "- A modality wins strongly if it retrieves the target much higher, or retrieves it while the other misses.\n"
            "- A modality wins weakly if the rank gap is small.\n"
            "- Prioritize consistent trends in memory over a single noisy observation.\n"
            "- If current evidence conflicts with memory, update conservatively unless the new evidence is very strong.\n\n"
            "Recall-size guidance:\n"
            "- Increase total_recall if both modalities miss, ranks are poor, or the query is broad/ambiguous.\n"
            "- Decrease total_recall if the dominant modality retrieves targets reliably at high ranks and the query is specific.\n"
            "- Keep total_recall stable when current evidence is mixed or the previous value seems adequate.\n"
            "- Do not only increase recall to compensate for a bad modality; reduce that modality's weight when confidence is high.\n\n"
            "Constraints:\n"
            "- text_weight and vl_weight must be floats in [0.0, 1.0].\n"
            "- text_weight + vl_weight should be 1.0.\n"
            "- total_recall must be a positive integer.\n"
            "- Obey any bounds provided in the input constraints, especially total_recall_range and max_step_weight_change.\n"
            "- If evidence is sufficient and you can determine a valid update confidently, output your own LLM decision.\n"
            "- If evidence is insufficient/uncertain or you cannot determine a reliable valid update, fall back to rule_hints "
            "(rule_next_text_weight, rule_next_vl_weight, rule_next_total_recall) as the output baseline.\n"
            "- Use rule_hints as guidance, but make the final decision with LLM reasoning.\n"
            "- Output strictly one valid JSON object only, with no markdown or extra text.\n\n"
            "JSON schema:\n"
            "{\n"
            '  "text_weight": float,\n'
            '  "vl_weight": float,\n'
            '  "total_recall": int,\n'
            '  "summary": string,\n'
            '  "reasoning": string\n'
            "}\n\n"
            "The reasoning field should briefly mention: "
            "which modality performed better, whether the query is text-driven, VL-driven, or mixed, "
            "how memory/rule_hints affected the update, and why total_recall changed or stayed stable.\n\n"
            f"Input:\n{json.dumps(context_payload, ensure_ascii=False)}"
        )
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=min(1024, int(self.max_new_tokens)),
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        try:
            index = len(output_ids) - output_ids[::-1].index(self._tokenizer.convert_tokens_to_ids("</think>"))
        except ValueError:
            index = 0
        content = self._tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        payload = self._try_json_decode(content)
        if payload is None:
            return None
        try:
            text_weight = float(payload.get("text_weight", current_text_weight))
            vl_weight = float(payload.get("vl_weight", 1.0 - text_weight))
            total_recall = int(payload.get("total_recall", current_total_recall))
        except Exception:
            return None
        text_weight = max(0.05, min(0.95, text_weight))
        vl_weight = max(0.05, min(0.95, vl_weight))
        weight_sum = text_weight + vl_weight
        if weight_sum <= 1e-8:
            text_weight, vl_weight = float(current_text_weight), float(current_vl_weight)
        else:
            text_weight /= weight_sum
            vl_weight = 1.0 - text_weight
        total_recall = max(recall_floor, min(recall_ceiling, total_recall))
        return {
            "text_weight": round(text_weight, 4),
            "vl_weight": round(vl_weight, 4),
            "total_recall": int(total_recall),
            "summary": str(payload.get("summary", "")).strip(),
            "reasoning": str(payload.get("reasoning", "")).strip(),
        }
