#!/usr/bin/env python3
"""Submit the medical agent as a Maze static workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from maze import MaClient

from src.utils.case_loader import load_case_input
from workflow import build_medical_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="通过 Maze 运行医学文本四任务流水线")
    parser.add_argument("case_path", type=Path, help="经授权的 UTF-8 病例文本")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--provider", default="example")
    parser.add_argument("--model", default="your-chat-model")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--api-key-file", type=Path, default=None, help="可选；仅传路径，不把密钥写入 Maze 输入")
    parser.add_argument("--disable-question-reflection", action="store_true")
    parser.add_argument("--artifact-mode", action="store_true")
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    case_path = args.case_path.expanduser().resolve()
    case_input = load_case_input(case_path)
    case_id = case_path.stem
    output_root = (args.output_dir / case_id).resolve()
    api_key_file = str(args.api_key_file.expanduser().resolve()) if args.api_key_file else ""

    delivery_root = project_root.parent
    manifest_path = delivery_root / "data" / "rag" / "manifest.json"
    chroma_path = delivery_root / "data" / "rag" / "local_chroma"
    if not manifest_path.is_file() or not chroma_path.is_dir():
        raise SystemExit("RAG 索引缺失；请将预建的只读索引（manifest.json 与 local_chroma/）放置于交付根目录 data/rag/，运行期不会自动重建。")

    client = MaClient(args.server_url)
    workflow, _ = build_medical_workflow(
        client,
        case_input=case_input,
        provider=args.provider,
        model=args.model,
        project_root=project_root,
        output_root=output_root,
        case_id=case_id,
        enable_question_reflection=not args.disable_question_reflection,
        api_key_file=api_key_file,
    )
    run_id = workflow.run(
        workspace_dir=str(project_root),
        artifact_mode=args.artifact_mode,
        timeout_seconds=args.timeout,
        tags=["medical-agent", "four-task"],
        metadata={"case_id": case_id, "pipeline": "medical_text_v2"},
    )
    snapshot = workflow.wait(run_id, timeout=args.timeout)
    results = workflow.show_results(run_id)
    print(json.dumps({"run_id": run_id, "snapshot": snapshot, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
