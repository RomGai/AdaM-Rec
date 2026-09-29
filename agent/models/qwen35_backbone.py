"""Shared Qwen3.5 generation model for text and image/text prompts.

Retrieval uses the existing dedicated text and multimodal embedding models.
"""

from functools import lru_cache

from agent.config.backbone_config import DEFAULT_BACKBONE


def load_backbone(model_name=DEFAULT_BACKBONE, dtype="auto", device="auto"):
    # Normalize calls with omitted/explicit defaults to a single cache key.
    return _load_backbone_cached(str(model_name), dtype, "auto" if device == "cuda" else device)


@lru_cache(maxsize=None)
def _load_backbone_cached(model_name, dtype, device):
    try:
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    except ImportError as exc:
        raise ImportError("Install transformers with Qwen3_5ForConditionalGeneration support (>=5.3.0).") from exc
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name, dtype=dtype, device_map=device,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def load_text_backbone(model_name=DEFAULT_BACKBONE):
    model, processor = load_backbone(model_name)
    return model, processor.tokenizer

