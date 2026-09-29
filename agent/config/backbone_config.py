"""Configure generation backbones independently from retrieval encoders."""

import argparse

DEFAULT_BACKBONE = "Qwen/Qwen3.5-9B"
GENERATION_FIELDS = ("backbone", "model", "text_model", "vl_model")


class BackboneAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        for field in GENERATION_FIELDS:
            setattr(namespace, field, values)


def add_backbone_argument(parser):
    """Text and image generation share one backbone; embedding flags are untouched."""
    parser.set_defaults(**dict.fromkeys(GENERATION_FIELDS, DEFAULT_BACKBONE))
    parser.add_argument(
        "--backbone", "--model", "--text-model", "--vl-model",
        dest="backbone", action=BackboneAction, default=DEFAULT_BACKBONE,
        help="Generation backbone for routing, profiling, reasoning and reranking; does not change embedding models.",
    )
