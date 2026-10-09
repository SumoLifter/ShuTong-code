"""Parse free-text medical case into structured_case dict.

Extracted and simplified from medical_agent/src/graph/nodes.py
"""
from __future__ import annotations

import re
from typing import Any


LABELS = [
    "患者基本信息",
    "患者信息",
    "基本信息",
    "主诉",
    "现病史",
    "既往史",
    "个人史",
    "家族史",
    "体格检查",
    "体格检查（入院查体）",
    "检查与查体",
    "入院查体",
    "专科查体",
    "辅助检查",
    "检查与化验",
    "辅助检查（化验与影像）",
    "诊断",
    "初步诊断",
    "出院诊断",
    "治疗方案",
    "治疗与手术",
    "治疗经过",
    "治疗经过（医嘱）",
    "治疗经过（根据现病史及既往史整理）",
    "手术及治疗经过",
    "治疗与医嘱",
    "医嘱/治疗经过",
    "医嘱/处理",
    "出院情况",
]


def _label_pattern(label: str) -> str:
    return rf"(?:[一二三四五六七八九十]+[、.．]\s*)?{re.escape(label)}"


_ALL_LABELS_PATTERN = "|".join(_label_pattern(item) for item in LABELS)


def _extract_section(text: str, label_pattern: str, all_labels_pattern: str = _ALL_LABELS_PATTERN) -> Any:
    label_pattern = _label_pattern(label_pattern)
    heading_prefix = r"(?:[-*]\s*)?(?:#{1,6}\s*)?(?:\*\*)?\s*"
    heading_suffix = r"\s*(?:\*\*)?\s*"
    next_heading = rf"\n\s*{heading_prefix}(?:{all_labels_pattern}){heading_suffix}(?:[：:]|\n|$)"
    patterns = [
        rf"(?:^|\n)\s*(?:#{1,6}\s*)+(?:{label_pattern}){heading_suffix}[：:]?\s*\n+(.*?)(?={next_heading}|\n\s*#{1,6}\s+|$)",
        rf"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)\s*(?:{label_pattern}){heading_suffix}[：:]?\s*\n+(.*?)(?={next_heading}|$)",
        rf"(?:^|\n)\s*{heading_prefix}(?:{label_pattern}){heading_suffix}[：:]\s*(.*?)(?={next_heading}|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.S)
        if match:
            value = match.group(1).strip()
            if value:
                return value
    return None


def _merge_sections(*values: Any) -> Any:
    merged = []
    seen = set()
    for item in values:
        if not item:
            continue
        normalized = str(item).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
    if not merged:
        return None
    return "\n".join(merged)


def _extract_heading_block(text: str, heading_pattern: str) -> Any:
    heading_pattern = _label_pattern(heading_pattern)
    next_heading = rf"\n\s*(?:[-*]\s*)?(?:#{1,6}\s*)?(?:\*\*)?\s*(?:{_ALL_LABELS_PATTERN})\s*(?:\*\*)?\s*(?:[：:]|\n|$)"
    patterns = [
        rf"(?:^|\n)\s*#+\s*[^\n]*?(?:{heading_pattern})\s*\n(.*?)(?={next_heading}|\n\s*#+\s+|$)",
        rf"(?:^|\n)\s*(?:\*\*)\s*(?:{heading_pattern})\s*(?:\*\*)\s*\n(.*?)(?={next_heading}|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.S)
        if match:
            value = match.group(1).strip()
            if value:
                return value
    return None


def _extract_key_value(block_text: str, key_pattern: str) -> Any:
    if not block_text:
        return None
    patterns = [
        rf"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?\s*(?:{key_pattern})\s*(?:\*\*)?\s*[：:]\s*([^\n]+)",
        rf"(?:^|\n)\s*(?:[-*]\s*)?\*\*\s*(?:{key_pattern})\s*[：:]\s*\*\*\s*([^\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, block_text)
        if not match:
            continue
        value = match.group(1).strip()
        if value.startswith("**") and value.endswith("**") and len(value) > 4:
            value = value[2:-2].strip()
        if value:
            return value
    return None


def parse_case(text: str) -> dict:
    """Parse raw case text into structured_case dict."""
    basic_info_block = _merge_sections(
        _extract_heading_block(text, "患者基本信息"),
        _extract_heading_block(text, "患者信息"),
        _extract_heading_block(text, "基本信息"),
    )

    chief_complaint = _merge_sections(_extract_heading_block(text, "主诉"), _extract_section(text, "主诉"))
    history = _merge_sections(_extract_heading_block(text, "现病史"), _extract_section(text, "现病史"))
    past_history = _merge_sections(_extract_heading_block(text, "既往史"), _extract_section(text, "既往史"))
    physical_exam = _merge_sections(
        _extract_section(text, "体格检查"),
        _extract_section(text, "检查与查体"),
        _extract_section(text, "体格检查（入院查体）"),
        _extract_section(text, "三、体格检查"),
        _extract_section(text, "入院查体"),
        _extract_heading_block(text, "体格检查（入院查体）"),
        _extract_heading_block(text, "三、体格检查"),
        _extract_heading_block(text, "体格检查"),
        _extract_heading_block(text, "入院查体"),
    )
    auxiliary_exam = _merge_sections(
        _extract_section(text, "辅助检查"),
        _extract_section(text, "检查与化验"),
        _extract_section(text, "辅助检查（化验与影像）"),
        _extract_heading_block(text, "辅助检查"),
        _extract_heading_block(text, "检查与化验"),
    )
    diagnosis = _merge_sections(
        _extract_heading_block(text, "诊断"),
        _extract_heading_block(text, "初步诊断"),
        _extract_heading_block(text, "出院诊断"),
        _extract_section(text, "初步诊断"),
        _extract_section(text, "出院诊断"),
    )
    treatment = _merge_sections(
        _extract_section(text, "治疗方案"),
        _extract_section(text, "治疗与手术"),
        _extract_section(text, "手术及治疗经过"),
        _extract_section(text, "治疗与医嘱"),
        _extract_section(text, "治疗经过"),
        _extract_section(text, "治疗经过（医嘱）"),
        _extract_section(text, "治疗经过（根据现病史及既往史整理）"),
        _extract_section(text, "医嘱/治疗经过"),
        _extract_section(text, "六、治疗经过"),
        _extract_section(text, "出院情况"),
        _extract_heading_block(text, "治疗经过"),
        _extract_heading_block(text, "治疗经过（医嘱）"),
        _extract_heading_block(text, "治疗经过（根据现病史及既往史整理）"),
        _extract_heading_block(text, "医嘱/治疗经过"),
        _extract_heading_block(text, "六、治疗经过"),
        _extract_heading_block(text, "治疗与手术"),
        _extract_heading_block(text, "出院情况"),
    )
    basic_info = {
        "name": _extract_key_value(basic_info_block, "姓名"),
        "gender": _extract_key_value(basic_info_block, "性别"),
        "age": _extract_key_value(basic_info_block, "年龄"),
        "department": _extract_key_value(basic_info_block, "科室"),
        "admission_date": _extract_key_value(basic_info_block, "入院日期"),
        "discharge_date": _extract_key_value(basic_info_block, "出院日期"),
    }
    return {
        "basic_info": basic_info,
        "chief_complaint": chief_complaint,
        "history": history,
        "past_history": past_history,
        "physical_exam": physical_exam,
        "auxiliary_exam": auxiliary_exam,
        "diagnosis": diagnosis,
        "treatment": treatment,
    }
