"""七维教学病例评分 Agent：JSON 为主，自动渲染 Markdown。"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.base_agent import BaseAgent


DIMENSIONS = [
    "病例资料完整度",
    "病例书写的规范性",
    "病例学习内容的丰富度",
    "病例典型性",
    "病例稀缺性",
    "诊疗思路的完整性",
    "MDT需求度",
]


class TeachingAlignmentScoringAgent(BaseAgent):
    """使用 teaching_alignment/v2 的七维 JSON 契约评分。

    评分只使用病例原文和结构化病例，不注入 RAG 或 reasoning。
    """

    def __init__(self, provider: str, model: str, temperature: float = 0.3):
        super().__init__("scoring", provider, model, temperature, use_rag=False)

    @staticmethod
    def _parse_json(content: str) -> Dict[str, Any]:
        text = re.sub(r"^\s*```(?:json)?\s*", "", content, flags=re.I)
        text = re.sub(r"\s*```\s*$", "", text).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("评分模型未返回合法 JSON")
            data = json.loads(text[start : end + 1])
        if not isinstance(data, dict):
            raise ValueError("评分输出必须是 JSON 对象")
        return data

    @staticmethod
    def _validate(data: Dict[str, Any], case_id: str) -> Dict[str, Any]:
        dimensions = data.get("dimensions")
        if not isinstance(dimensions, list) or len(dimensions) != len(DIMENSIONS):
            raise ValueError("评分输出必须包含刚好 7 个维度")

        normalized: List[Dict[str, Any]] = []
        for expected, item in zip(DIMENSIONS, dimensions):
            if not isinstance(item, dict) or item.get("dimension") != expected:
                raise ValueError(f"评分维度顺序或名称错误，期望：{expected}")
            score = item.get("score")
            if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 10:
                raise ValueError(f"{expected} 的 score 必须为 0-10 整数")
            normalized.append(
                {
                    "dimension": expected,
                    "score": score,
                    "evidence": str(item.get("evidence", "")).strip(),
                    "reason": str(item.get("reason", "")).strip(),
                    "suggestion": str(item.get("suggestion", "")).strip(),
                }
            )

        confidence = data.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, int) or not 0 <= confidence <= 100:
            raise ValueError("confidence 必须为 0-100 整数")
        return {
            "case_id": str(data.get("case_id") or case_id),
            "dimensions": normalized,
            "overall_comment": str(data.get("overall_comment", "")).strip(),
            "confidence": confidence,
        }

    @staticmethod
    def render_markdown(data: Dict[str, Any]) -> str:
        lines = [
            "# 教学病例七维评分报告",
            "",
            f"- **病例标识**：{data['case_id']}",
            f"- **评分置信度**：{data['confidence']}/100",
            "",
            "## 维度评分",
            "",
            "| 维度 | 分数 |",
            "| --- | ---: |",
        ]
        lines.extend(f"| {item['dimension']} | {item['score']}/10 |" for item in data["dimensions"])
        for index, item in enumerate(data["dimensions"], 1):
            lines.extend(
                [
                    "",
                    f"## {index}. {item['dimension']}（{item['score']}/10）",
                    f"**病例依据**：{item['evidence'] or '未提供'}",
                    "",
                    f"**评分理由**：{item['reason'] or '未提供'}",
                    "",
                    f"**改进建议**：{item['suggestion'] or '未提供'}",
                ]
            )
        lines.extend(["", "## 总体评价", data["overall_comment"] or "未提供", ""])
        return "\n".join(lines)

    def execute(self, structured_case: dict, case_text: str = "", case_flags: dict | None = None) -> str:
        prompt_text = self.prompt_loader.load_optional_text("scoring/executor.txt")
        payload = {
            "case_id": str((case_flags or {}).get("case_id", "")),
            # 用户决定：原始病例直接作为评分对齐文本，不做全局清洗流程。
            "clean_case_markdown": case_text,
            "structured_case": structured_case,
        }
        response = self.llm_client.chat(
            messages=[
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            ]
        )
        error = self._is_llm_error_content(response.content)
        if error:
            raise RuntimeError(error)
        return json.dumps(self._validate(self._parse_json(self._strip_think(response.content)), payload["case_id"]), ensure_ascii=False, indent=2)

    def run(self, structured_case: dict, output_dir: Optional[Path] = None, filename: Optional[str] = None, **execute_kwargs) -> Dict[str, Any]:
        started = datetime.now()
        parsed_case, raw_text, case_flags = self._split_case_input(structured_case)
        case_id = str(case_flags.get("case_id") or filename or "")
        case_flags = {**case_flags, "case_id": case_id}
        try:
            content = self.execute(parsed_case, case_text=raw_text, case_flags=case_flags, **execute_kwargs)
            data = json.loads(content)
            status, error = "completed", None
        except Exception as exc:
            content, data, status, error = "", None, "failed", str(exc)

        result = {
            "task": self.task_name,
            "status": status,
            "content": content,
            "error": error,
            "started_at": started.isoformat(timespec="seconds"),
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "provider": self.provider,
            "model": self.model,
        }
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "scoring_meta.json").write_text(
                json.dumps({k: v for k, v in result.items() if k != "content"}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if data is not None:
                (output_dir / "scoring.json").write_text(content + "\n", encoding="utf-8")
                (output_dir / "scoring.md").write_text(self.render_markdown(data), encoding="utf-8")
            else:
                (output_dir / "scoring.md").write_text(f"<!-- ERROR: {error} -->\n", encoding="utf-8")
        return result
