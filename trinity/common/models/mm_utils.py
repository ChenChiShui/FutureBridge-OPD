""""Multi-modal utilities for processing and handling multi-modal data such as images and videos.
Only support Qwen2.5 VL series.

Modified from: verl/utils/dataset/rl_dataset.py
"""
import re
from typing import Any, Dict, List

import numpy as np
from PIL import Image


def build_multi_modal_inputs(
    prompt: str,
    images: List[Image.Image],
    videos: List[np.ndarray],
    processor: Any,
) -> Dict[str, Any]:
    """
    Preprocess multi-modal data and build multi-modal inputs
    """
    if prompt is None:
        raise ValueError("Prompt is required for build multi-modal inputs")

    multi_modal_data = {}
    if images:
        multi_modal_data["image"] = images
    if videos:
        multi_modal_data["video"] = videos

    model_inputs = processor(
        text=[prompt],
        images=multi_modal_data.get("image", None),
        videos=multi_modal_data.get("video", None),
        return_tensors="pt",
    )

    input_ids = model_inputs.pop("input_ids")[0]
    model_inputs.pop("attention_mask")

    if "second_per_grid_ts" in model_inputs:
        model_inputs.pop("second_per_grid_ts")

    return {
        "prompt": prompt,
        "prompt_token_ids": input_ids,
        "multi_modal_data": multi_modal_data,
        "multi_modal_inputs": dict(model_inputs),
    }


def convert_messages_to_mm_format(messages: List[Dict]) -> List[Dict]:
    for message in messages:
        content = message["content"]
        content_list = []
        segments = re.split("(<image>|<video>)", content)
        segments = [item for item in segments if item != ""]
        for segment in segments:
            if segment == "<image>":
                content_list.append(
                    {"type": "image"}
                )  # chat template will fill the actual image data later
            elif segment == "<video>":
                content_list.append(
                    {"type": "video"}
                )  # chat template will fill the actual video data later
            elif len(segment) == 0:
                continue
            else:
                content_list.append({"type": "text", "text": segment})

        message["content"] = content_list
    return messages
