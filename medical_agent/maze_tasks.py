"""Maze task definitions for the four-task medical workflow."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from maze import task


def _activate_project(project_root: str, api_key_file: str = "") -> Path:
    root = Path(project_root).expanduser().resolve()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    if api_key_file:
        key_path = Path(api_key_file).expanduser().resolve()
        raw = key_path.read_text(encoding="utf-8-sig").strip()
        if "=" in raw and "\n" not in raw:
            _, raw = raw.split("=", 1)
            raw = raw.strip().strip('"').strip("'")
        if raw:
            os.environ.setdefault("LLM_API_KEY", raw)
    return root


def _result_fields(result: dict[str, Any], task_name: str) -> dict[str, str]:
    return {
        "task": task_name,
        "status": str(result.get("status") or "failed"),
        "content": str(result.get("content") or ""),
        "error": str(result.get("error") or ""),
    }


@task(task_kind="io", resources={"cpu_num": 1, "io_num": 1}, timeout_seconds=900)
def scoring_task(
    case_input: dict,
    provider: str,
    model: str,
    project_root: str,
    output_root: str,
    case_id: str,
    api_key_file: str = "",
):
    _activate_project(project_root, api_key_file)
    from src.agents import TeachingAlignmentScoringAgent

    result = TeachingAlignmentScoringAgent(provider, model).run(
        case_input,
        output_dir=Path(output_root) / "scoring",
        filename=case_id,
    )
    fields = _result_fields(result, "scoring")
    return {"task": fields["task"], "status": fields["status"], "content": fields["content"], "error": fields["error"]}


@task(task_kind="io", resources={"cpu_num": 1, "io_num": 1}, timeout_seconds=900)
def soap_task(
    case_input: dict,
    provider: str,
    model: str,
    project_root: str,
    output_root: str,
    case_id: str,
    api_key_file: str = "",
):
    _activate_project(project_root, api_key_file)
    from src.agents import SOAPTextAgent

    result = SOAPTextAgent(provider, model, use_rag=True).run(
        case_input,
        output_dir=Path(output_root) / "soap",
        filename=case_id,
    )
    fields = _result_fields(result, "soap")
    return {"task": fields["task"], "status": fields["status"], "content": fields["content"], "error": fields["error"]}


@task(task_kind="io", resources={"cpu_num": 1, "io_num": 1}, timeout_seconds=900)
def reasoning_task(
    case_input: dict,
    provider: str,
    model: str,
    project_root: str,
    output_root: str,
    case_id: str,
    api_key_file: str = "",
):
    _activate_project(project_root, api_key_file)
    from src.agents import ReasoningTextAgent

    result = ReasoningTextAgent(provider, model, use_rag=True).run(
        case_input,
        output_dir=Path(output_root) / "reasoning",
        filename=case_id,
    )
    fields = _result_fields(result, "reasoning")
    return {"task": fields["task"], "status": fields["status"], "content": fields["content"], "error": fields["error"]}


@task(task_kind="io", resources={"cpu_num": 1, "io_num": 1}, timeout_seconds=1800)
def question_task(
    case_input: dict,
    soap_content: str,
    soap_status: str,
    reasoning_content: str,
    reasoning_status: str,
    provider: str,
    model: str,
    project_root: str,
    output_root: str,
    case_id: str,
    enable_reflection: bool = True,
    api_key_file: str = "",
):
    _activate_project(project_root, api_key_file)
    from src.agents import QuestionAgent

    degraded = []
    if soap_status != "completed":
        soap_content = ""
        degraded.append("soap")
    if reasoning_status != "completed":
        reasoning_content = ""
        degraded.append("reasoning")
    result = QuestionAgent(
        provider,
        model,
        use_rag=True,
        enable_reflection=enable_reflection,
    ).run(
        case_input,
        output_dir=Path(output_root) / "question",
        filename=case_id,
        soap_content=soap_content,
        reasoning_content=reasoning_content,
    )
    meta_path = Path(output_root) / "question" / f"{case_id}_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["upstream_status"] = {"soap": soap_status, "reasoning": reasoning_status}
        meta["degraded_inputs"] = degraded
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = _result_fields(result, "question")
    return {
        "task": fields["task"],
        "status": fields["status"],
        "content": fields["content"],
        "error": fields["error"],
        "degraded_inputs": degraded,
    }


@task(task_kind="io", resources={"cpu_num": 1, "io_num": 1}, timeout_seconds=120)
def finalize_task(
    scoring_status: str,
    soap_status: str,
    reasoning_status: str,
    question_status: str,
    question_degraded_inputs: list,
    output_root: str,
    case_id: str,
):
    statuses = {
        "scoring": scoring_status,
        "soap": soap_status,
        "reasoning": reasoning_status,
        "question": question_status,
    }
    overall = "completed" if all(value == "completed" for value in statuses.values()) else "degraded"
    summary = {
        "case_id": case_id,
        "status": overall,
        "tasks": statuses,
        "question_degraded_inputs": list(question_degraded_inputs or []),
        "output_dir": str(Path(output_root).resolve()),
    }
    summary_path = Path(output_root) / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": overall, "summary": summary, "summary_path": str(summary_path.resolve())}
