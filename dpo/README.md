# DPO training

`dpo/main.py` contains the existing training implementation. The public root
command is:

```bash
python dpo_main.py --dataset <private-dpo-jsonl> --model-name <model-path> --output-dir output/dpo_model
```

The loader accepts the original JSONL schema (`prompt`, `chosen`, `rejected`),
uses every preference row, and does not create an automatic validation split.
LoRA is the default; `--use-qlora` and `--full-finetune` retain the existing
training choices. Checkpoints and resume are supported by the existing
implementation.

Install `dpo/requirements_gpu.txt` in a separate CUDA environment. The public
repository does not include model weights or training data.
