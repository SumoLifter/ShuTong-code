# -*- coding: utf-8 -*-
"""DPO 训练所需的最小模型工具集。"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

LOCAL_MODEL_DIR = Path(os.environ.get("AIEDU_MODEL_CACHE_DIR", "resources/model"))


def resolve_model_path(model_path: str) -> str:
    """解析模型路径：本地存在则直接返回，否则按 HuggingFace repo_id 下载到本地缓存。"""

    def normalize_baichuan_name(hf_name: str) -> str:
        baichuan_mapping = {
            "Baichuan-M2-32B-0226": "baichuan-inc/Baichuan-M2-32B",
            "Baichuan-M2-32B": "baichuan-inc/Baichuan-M2-32B",
            "baichuan-m2-32b": "baichuan-inc/Baichuan-M2-32B",
            "Baichuan-M2-7B-0226": "baichuan-inc/Baichuan-M2-7B",
            "Baichuan-M2-7B": "baichuan-inc/Baichuan-M2-7B",
        }
        return baichuan_mapping.get(hf_name.strip(), hf_name)

    candidate = Path(model_path).expanduser()
    if candidate.exists():
        logger.info("使用本地模型: %s", candidate)
        return str(candidate)

    # A missing path containing a separator is treated as a requested local
    # destination only when it is clearly a path.  A Hugging Face repo id such
    # as org/model remains a remote identifier and is cached below.
    is_probable_local = (
        candidate.is_absolute()
        or model_path.startswith(("./", "../"))
        or model_path.startswith(("resources/", "output/", "data/"))
        or "\\" in model_path
    )
    if is_probable_local:
        dir_name = candidate.name
        hf_name = normalize_baichuan_name(dir_name.replace("__", "/"))
        logger.info("本地模型不存在，将下载到指定路径: %s", candidate)
        _download_to_local(hf_name, str(candidate))
        return str(candidate)

    hf_name_normalized = normalize_baichuan_name(model_path)
    local_path = LOCAL_MODEL_DIR / hf_name_normalized.replace("/", "__")
    if local_path.exists():
        logger.info("使用本地模型: %s", local_path)
        return str(local_path)

    logger.info("本地模型不存在，将下载: %s -> %s", hf_name_normalized, local_path)
    _download_to_local(hf_name_normalized, str(local_path))
    return str(local_path)


def _download_to_local(hf_name: str, local_path: str) -> None:
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parent_dir = os.path.dirname(local_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    logger.info("正在从 HuggingFace 下载模型: %s -> %s", hf_name, local_path)

    try:
        snapshot_download(
            repo_id=hf_name,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        logger.info("模型下载完成: %s", local_path)
    except Exception as exc:
        logger.warning("snapshot_download 失败: %s，尝试 from_pretrained", exc)
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(hf_name, trust_remote_code=True)
        tokenizer.save_pretrained(local_path)
        model.save_pretrained(local_path)
        logger.info("模型下载并保存完成: %s", local_path)


def detect_runtime_device() -> str:
    """Detect the training device.  The supported accelerator is CUDA."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


__all__ = [
    "resolve_model_path",
    "detect_runtime_device",
]
