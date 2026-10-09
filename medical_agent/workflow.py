"""Maze DAG construction for the medical agent."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from maze import MaClient

from maze_tasks import finalize_task, question_task, reasoning_task, scoring_task, soap_task


def build_medical_workflow(
    client: MaClient,
    *,
    case_input: dict[str, Any],
    provider: str,
    model: str,
    project_root: Path,
    output_root: Path,
    case_id: str,
    enable_question_reflection: bool = True,
    api_key_file: str = "",
):
    workflow = client.create_workflow()
    common = {
        "case_input": case_input,
        "provider": provider,
        "model": model,
        "project_root": str(project_root.resolve()),
        "output_root": str(output_root.resolve()),
        "case_id": case_id,
        "api_key_file": api_key_file,
    }
    scoring = workflow.add_task(scoring_task, inputs=common, task_name="medical_scoring")
    soap = workflow.add_task(soap_task, inputs=common, task_name="medical_soap")
    reasoning = workflow.add_task(reasoning_task, inputs=common, task_name="medical_reasoning")
    question = workflow.add_task(
        question_task,
        inputs={
            **common,
            "soap_content": soap.outputs["content"],
            "soap_status": soap.outputs["status"],
            "reasoning_content": reasoning.outputs["content"],
            "reasoning_status": reasoning.outputs["status"],
            "enable_reflection": enable_question_reflection,
        },
        task_name="medical_question",
    )
    final = workflow.add_task(
        finalize_task,
        inputs={
            "scoring_status": scoring.outputs["status"],
            "soap_status": soap.outputs["status"],
            "reasoning_status": reasoning.outputs["status"],
            "question_status": question.outputs["status"],
            "question_degraded_inputs": question.outputs["degraded_inputs"],
            "output_root": str(output_root.resolve()),
            "case_id": case_id,
        },
        task_name="medical_finalize",
    )
    return workflow, {
        "scoring": scoring,
        "soap": soap,
        "reasoning": reasoning,
        "question": question,
        "finalize": final,
    }
