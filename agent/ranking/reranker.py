"""LLM-based item reranker for Amazon recommendation (Qwen3.5-9B).

Agent 5 scoring:
- 5-level relevance buckets (1~5)
- logits-based weighted expectation score

Scores text representations of product profiles against dynamic user constraints.
"""

from __future__ import annotations

from agent.models.qwen35_backbone import load_text_backbone


import json
from typing import Any, Dict, List

try:
    import torch
except Exception:  # pragma: no cover
    torch = None




class LLMItemReranker:
    """Rerank candidate items with Qwen3.5-9B via five-level logits weighting."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-9B",
        enable_thinking: bool = False,
    ) -> None:
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self._tokenizer = None
        self._model = None

        self.id_1 = None
        self.id_2 = None
        self.id_3 = None
        self.id_4 = None
        self.id_5 = None

    def load(self) -> None:
        if self._model is None:
            self._model, self._tokenizer = load_text_backbone(self.model_name)
        for level in range(1, 6):
            ids = self._tokenizer.encode(str(level), add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError("Relevance levels 1..5 must each be a single token.")
            setattr(self, f"id_{level}", ids[0])

    @(torch.no_grad() if torch is not None else (lambda function: function))
    def _score_with_logits(self, prompt: str) -> Dict[str, Any]:
        self.load()
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        logits = self._model(**inputs, logits_to_keep=1, use_cache=False).logits[:, -1, :]

        score_logits = torch.stack(
            [
                logits[:, self.id_1],
                logits[:, self.id_2],
                logits[:, self.id_3],
                logits[:, self.id_4],
                logits[:, self.id_5],
            ],
            dim=1,
        )
        probs = torch.nn.functional.softmax(score_logits, dim=1)[0]
        p = probs.tolist()
        weighted = sum((idx + 1) * val for idx, val in enumerate(p))

        return {
            "probs": {"1": p[0], "2": p[1], "3": p[2], "4": p[3], "5": p[4]},
            "weighted_score": float(weighted),
        }

    @staticmethod
    def _build_scoring_prompt(
        query: str,
        preference_constraints: Dict[str, Any],
        item: Dict[str, Any],
    ) -> str:
        must_have = preference_constraints.get("Must_Have", [])
        nice_to_have = preference_constraints.get("Nice_to_Have", [])
        must_avoid = preference_constraints.get("Must_Avoid", [])

        profile = item.get("profile", {})
        compact_item = {
            "item_id": item.get("item_id"),
            "title": profile.get("title", ""),
            "taxonomy": profile.get("taxonomy", {}),
            "text_tags": profile.get("text_tags", {}),
            "visual_tags": profile.get("visual_tags", {}),
            "hypotheses": profile.get("hypotheses", []),
            "overall_confidence": profile.get("overall_confidence", 0.0),
        }

        next_predictions = preference_constraints.get("Predicted_Next_Items", [])

        return (
            "你是电商推荐精排专家（Agent5）。请从用户视角判断候选商品与当下偏好的匹配程度。\n"
            "评分规则（只能取1~5）：\n"
            "注意：这是 next-item 预测场景，需重点判断该候选是否与 Predicted_Next_Items 一致。\n"
            "1=触碰Must_Avoid或与核心诉求明显冲突；\n"
            "2=弱相关，仅少量满足；\n"
            "3=中等相关，满足部分Must_Have或多个Nice_to_Have；\n"
            "4=高相关，满足大多数Must_Have且有Nice_to_Have加分；\n"
            "5=强匹配，完整满足Must_Have且无冲突，同时在Nice_to_Have表现突出。\n"
            "请只输出一个数字：1/2/3/4/5。\n\n"
            f"用户Query: {query}\n"
            f"Must_Have: {json.dumps(must_have, ensure_ascii=False)}\n"
            f"Nice_to_Have: {json.dumps(nice_to_have, ensure_ascii=False)}\n"
            f"Must_Avoid: {json.dumps(must_avoid, ensure_ascii=False)}\n"
            f"Predicted_Next_Items: {json.dumps(next_predictions, ensure_ascii=False)}\n"
            f"候选商品画像: {json.dumps(compact_item, ensure_ascii=False)}\n"
        )

    @staticmethod
    def _must_avoid_filter(preference_constraints: Dict[str, Any], item: Dict[str, Any]) -> bool:
        must_avoid = [str(x).strip().lower() for x in preference_constraints.get("Must_Avoid", []) if str(x).strip()]
        if not must_avoid:
            return False

        profile = item.get("profile", {})
        haystacks = [
            profile.get("title", ""),
            json.dumps(profile.get("taxonomy", {}), ensure_ascii=False),
            json.dumps(profile.get("text_tags", {}), ensure_ascii=False),
            json.dumps(profile.get("visual_tags", {}), ensure_ascii=False),
            json.dumps(profile.get("hypotheses", []), ensure_ascii=False),
        ]
        item_text = "\n".join(str(x) for x in haystacks).lower()
        return any(token and token in item_text for token in must_avoid)

    def rerank_items(
        self,
        query: str,
        preference_constraints: Dict[str, Any],
        candidate_items: List[Dict[str, Any]],
        top_n: int = 40,
    ) -> List[Dict[str, Any]]:
        self.load()
        if top_n <= 0:
            return []

        scored: List[Dict[str, Any]] = []
        for item in candidate_items:
            if self._must_avoid_filter(preference_constraints, item):
                continue

            prompt = self._build_scoring_prompt(query, preference_constraints, item)
            score_info = self._score_with_logits(prompt)
            enriched = dict(item)
            enriched["llm_weighted_score"] = score_info["weighted_score"]
            enriched["ranking_score"] = float(score_info["weighted_score"])
            enriched["score_probs"] = score_info["probs"]
            scored.append(enriched)

        scored.sort(
            key=lambda x: (
                float(x.get("ranking_score", 0.0)),
                float((x.get("score_probs") or {}).get("5", 0.0)),
            ),
            reverse=True,
        )

        final_items: List[Dict[str, Any]] = []
        for rank, row in enumerate(scored[:top_n], start=1):
            row["rank"] = rank
            final_items.append(row)
        return final_items
