# -*- coding: utf-8 -*-
"""项目主入口：DPO 训练。"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


# ``python dpo/main.py`` is the documented entry point.  In script mode
# Python puts ``dpo/`` (rather than the project root) on ``sys.path``; add the
# root so the same package imports also work with ``python -m dpo.main``.
if __package__ in {None, ""}:
    _project_root = str(Path(__file__).resolve().parents[1])
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

_training_log = os.environ.get("AIEDU_TRAINING_LOG", "scratch/training.log") or "scratch/training.log"
Path(_training_log).expanduser().parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(_training_log, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger(__name__)


def setup_env(enable_cache: bool = False) -> str:
    if os.environ.get("AIEDU_HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = os.environ["AIEDU_HF_ENDPOINT"]
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["AIEDU_ENABLE_CACHE"] = "1" if enable_cache else "0"

    cache_dir = os.path.abspath(os.environ.get("AIEDU_MODEL_CACHE_DIR", "./resources/cache"))
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    return cache_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aiedu：DPO 训练入口")

    parser.add_argument("--dataset", type=str, required=True, help="DPO 偏好数据路径（JSONL，字段：prompt, chosen, rejected, meta）")
    parser.add_argument("--model-name", type=str, required=True, help="基础模型名称或本地路径（必填，由使用者指定）")
    parser.add_argument("--tokenizer-name", type=str, default=None, help="Tokenizer 名称或本地路径（默认同 model-name）")
    parser.add_argument("--adapter-path", type=str, default=None, help="已有 LoRA/QLoRA adapter 路径；传入后按 PEFT 模型继续训练")
    parser.add_argument("--output-dir", type=str, default="output/dpo_model", help="模型输出目录")

    parser.add_argument("--num-epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=1, help="每设备 batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="权重衰减")
    parser.add_argument("--warmup-steps", type=int, default=0, help="warmup 步数")
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine", help="学习率调度器")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    parser.add_argument("--max-length", type=int, default=4096, help="prompt+completion 最大 token 长度")
    parser.add_argument("--max-prompt-length", type=int, default=3072, help="prompt 最大 token 长度")
    parser.add_argument("--train-samples", type=int, default=-1, help="训练样本数，-1 表示全部")

    parser.add_argument("--logging-steps", type=int, default=10, help="日志间隔")
    parser.add_argument("--save-steps", type=int, default=100, help="保存间隔")
    parser.add_argument("--save-total-limit", type=int, default=2, help="最多保留 checkpoint 数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume", action="store_true", help="从 output-dir 下最近的 checkpoint 续跑；无 checkpoint 时从头开始")

    parser.add_argument("--full-finetune", action="store_true", help="关闭 LoRA，改为全参数 DPO")
    parser.add_argument("--use-qlora", action="store_true", help="在 CUDA 上启用 4bit QLoRA")
    parser.add_argument("--use-gradient-checkpointing", type=lambda x: x.lower() in ("true", "1", "yes"), default=True)
    parser.add_argument("--enable-cache", action="store_true", help="启用运行时缓存")

    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="LoRA target modules，逗号分隔",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> DPOTrainConfig:
    try:
        from dpo.utils.dpo import DPOTrainConfig
    except ModuleNotFoundError:
        from .utils.dpo import DPOTrainConfig

    return DPOTrainConfig(
        model_name_or_path=args.model_name,
        tokenizer_name_or_path=args.tokenizer_name,
        adapter_path=args.adapter_path,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        max_samples=args.train_samples,
        beta=args.beta,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        resume=args.resume,
        enable_cache=args.enable_cache,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        full_finetune=args.full_finetune,
        use_qlora=args.use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )


def main() -> int:
    args = parse_args()
    # Keep ``--help`` and argument inspection available without importing the
    # GPU training stack.  The full environment is required only after a run
    # is actually requested.
    import torch

    try:
        from dpo.utils.dpo import train_dpo
        from dpo.utils.model import detect_runtime_device
    except ModuleNotFoundError:
        from .utils.dpo import train_dpo
        from .utils.model import detect_runtime_device

    config = build_config(args)

    cache_dir = setup_env(enable_cache=config.enable_cache)
    device = detect_runtime_device()

    logger.info("运行设备: %s", device)
    if device == "cuda":
        logger.info("CUDA 数量: %s", torch.cuda.device_count())
    logger.info("缓存目录: %s", cache_dir)
    logger.info("参数: %s", vars(args))

    try:
        train_dpo(config)
        logger.info("DPO 训练完成")
        return 0
    except KeyboardInterrupt:
        logger.warning("用户中断")
        return 130
    except Exception as exc:
        logger.exception("执行失败: %s", exc)
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
