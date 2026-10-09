"""统一病例加载：原文为主输入，结构化解析为辅助。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.utils.case_parser import parse_case


def _count_structured_chars(structured_case: dict) -> int:
    total = 0
    for value in structured_case.values():
        if isinstance(value, dict):
            total += sum(len(str(item or "")) for item in value.values())
        else:
            total += len(str(value or ""))
    return total


def _has_core_content(structured_case: dict) -> bool:
    core_fields = [
        "chief_complaint",
        "history",
        "past_history",
        "physical_exam",
        "auxiliary_exam",
        "diagnosis",
        "treatment",
    ]
    return any(str(structured_case.get(field) or "").strip() for field in core_fields)


def _looks_non_patient_record(raw_text: str) -> bool:
    markers = [
        "并未包含具体的患者病历信息",
        "教学病例",
        "实习记录",
        "本月工作反思",
        "带教老师点评",
        "研究生导师点评",
    ]
    return any(marker in raw_text for marker in markers)


def build_case_input(raw_text: str, case_id: str = "") -> Dict[str, Any]:
    structured_case = parse_case(raw_text)
    structured_chars = _count_structured_chars(structured_case)
    is_empty_structured = not _has_core_content(structured_case)
    return {
        "case_id": case_id,
        "raw_text": raw_text,
        "case_text": raw_text,
        "structured_case": structured_case,
        "case_flags": {
            "is_empty_structured": is_empty_structured,
            "looks_non_patient_record": is_empty_structured and _looks_non_patient_record(raw_text),
            "raw_chars": len(raw_text),
            "structured_chars": structured_chars,
        },
    }


def load_case_input(case_path: Path) -> Dict[str, Any]:
    raw_text = case_path.read_text(encoding="utf-8")
    return build_case_input(raw_text, case_path.stem)
