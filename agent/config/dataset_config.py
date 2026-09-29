"""Dataset presets for the shared dynamic-recall pipeline."""

import argparse
import re


DATASET_INPUTS = {
    "beauty": ("data/amazon_beauty/query_data1.csv", "data/amazon_beauty/meta_Beauty.filtered.jsonl"),
    "clothing": ("data/amazon_clothing/query_data1.csv", "data/amazon_clothing/meta_Clothing_Shoes_and_Jewelry.filtered.jsonl"),
    "music": ("data/amazon_music/query_data1.csv", "data/amazon_music/meta_CDs_and_Vinyl.filtered.jsonl"),
}

COMMON_DEFAULTS = {
    "keyword_recall_topk": 50,
    "enable_agent3_qwen3vl_embedding": True,
    "enable_agent3_adaptive_weighting": True,
    "agent3_adaptive_max_pseudo_queries": 5,
    "enable_llm_routing": True,
    "enable_collaborative_signal": True,
    "collaborative_similarity_threshold": 0.5,
    "agent3_pseudo_query_llm_rewrite": True,
    "recall_only": False,
    "skip_agent3_qwen3vl_cache_repair": True,
    "enable_vl_profiling": False,
}

DATASET_DEFAULTS = {
    "beauty": dict(agent3_adaptive_llm_recall_control=True,
                   agent3_adaptive_min_total_recall=400, agent3_adaptive_max_total_recall=500,
                   agent3_query_recall_pool="filtered", agent3_qwen3vl_chunk_size=50),
    "clothing": dict(agent3_adaptive_llm_recall_control=True,
                     agent3_adaptive_min_total_recall=400, agent3_adaptive_max_total_recall=600,
                     agent3_query_recall_pool="full", agent3_qwen3vl_chunk_size=50),
    "music": dict(agent3_adaptive_llm_recall_control=False,
                  agent3_adaptive_min_total_recall=600, agent3_adaptive_max_total_recall=800,
                  agent3_query_recall_pool="full", agent3_qwen3vl_chunk_size=25),
}


def _infer_dataset(path):
    if not path:
        return None
    parts = str(path).replace("\\", "/").lower().split("/")
    for part in reversed(parts):
        tokens = set(re.split(r"[^a-z0-9]+", part))
        if tokens & {"clothing", "cloth"}:
            return "clothing"
        if "beauty" in tokens:
            return "beauty"
        if "music" in tokens or {"cds", "vinyl"} <= tokens:
            return "music"
    return None


def apply_dataset_defaults(args):
    dataset = getattr(args, "dataset", "auto")
    if dataset == "auto":
        inferred = {_infer_dataset(getattr(args, key, None))
                    for key in ("query_csv", "filtered_meta_jsonl")} - {None}
        if len(inferred) > 1:
            raise ValueError("query-csv and filtered-meta-jsonl identify different datasets; check the input paths")
        dataset = next(iter(inferred), "beauty")
    if dataset not in DATASET_DEFAULTS:
        raise ValueError(f"Unknown dataset: {dataset}")
    args.dataset = dataset
    query, meta = DATASET_INPUTS[dataset]
    defaults = dict(COMMON_DEFAULTS, **DATASET_DEFAULTS[dataset], query_csv=query, filtered_meta_jsonl=meta)
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    if not 1 <= args.agent3_adaptive_min_total_recall <= args.agent3_adaptive_max_total_recall:
        raise ValueError("adaptive recall bounds must satisfy 1 <= min <= max")
    if args.agent3_qwen3vl_chunk_size < 1:
        raise ValueError("agent3-qwen3vl-chunk-size must be positive")
    return args


class DatasetArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        try:
            return apply_dataset_defaults(parsed)
        except ValueError as exc:
            self.error(str(exc))
