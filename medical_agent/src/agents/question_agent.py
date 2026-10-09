"""最终题目 Agent：三段生成、per-stage 反思和选择性修复。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Set

from src.agents.base_agent import BaseAgent


BATCH_TYPES = ["A1", "A2", "A3"]

QTYPE_REQUIREMENTS = {
    "A1": "1个单句题干、5个选项、1个答案；考查单一知识点；不得使用共享题干或兜底选项。",
    "A2": "1个病例摘要题干、5个选项、1个答案；考查临床分析；不得使用共享题干或兜底选项。",
    "A3": "1个共享题干、2-3个独立问题；各题考点不同；不得使用兜底选项。",
    "A4": "1个共享病例情景、3-6个递进问题；不得使用兜底选项。",
    "B1": "5个备选答案和严格5个问题；每个答案必须为 A-E 字母；不得使用兜底选项。",
}


class QuestionAgent(BaseAgent):
    """唯一保留的 Question 实现。

    初次生成可使用病例原文、SOAP、reasoning 与 RAG。修复阶段严格
    只使用 structured_case、SOAP、RAG、旧题块、few-shot 和问题清单。
    """

    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float = 0.3,
        use_few_shot: bool = True,
        use_rag: bool = True,
        enable_reflection: bool = True,
    ):
        super().__init__("question", provider, model, temperature, use_rag=use_rag)
        self.use_few_shot = use_few_shot
        self.enable_reflection = enable_reflection

    def _load_few_shot(self, qtype: str) -> str:
        if not self.use_few_shot:
            return ""
        return (Path(__file__).resolve().parents[2] / "prompts" / "question" / "few_shots" / f"{qtype}.md").read_text(encoding="utf-8")

    @staticmethod
    def _extract_qtype_block(content: str, qtype: str) -> str:
        match = re.search(rf"## {re.escape(qtype)}型题.*?(?=\n## |\Z)", content, re.S)
        return match.group(0) if match else ""

    def _rule_check(self, content: str, qtypes: List[str]) -> List[Dict[str, str]]:
        issues: List[Dict[str, str]] = []
        for qtype in qtypes:
            block = self._extract_qtype_block(content, qtype)
            if not block:
                issues.append({"qtype": qtype, "text": f"缺少 {qtype} 型题内容"})
                continue
            if qtype in {"A1", "A2"}:
                stem = re.search(r"\*\*题干\*\*[:：]\s*(.+?)(?=\n[A-E]\.\s|\Z)", block, re.S)
                if stem and re.search(r"[A-E]\.\s+\S+", stem.group(1)):
                    issues.append({"qtype": qtype, "text": f"{qtype} 题干中嵌入选项内容"})
            if qtype == "B1":
                invalid = [answer.strip() for answer in re.findall(r"\*\*答案\*\*[:：]\s*(.+)", block) if not re.fullmatch(r"[A-E]", answer.strip())]
                if invalid:
                    issues.append({"qtype": qtype, "text": "B1 答案必须是 A-E 单个字母"})
            if any(token in block for token in ("以上全部", "以上都对", "以上都是")):
                issues.append({"qtype": qtype, "text": f"{qtype} 存在兜底选项"})
        return issues

    def _invoke_critic(self, output: str, qtypes: List[str]) -> str:
        original_temperature = self.llm_client.temperature
        self.llm_client.temperature = 0.1
        try:
            return self._invoke("question/executor_critic.md", {"output": output, "requested_types": qtypes})
        finally:
            self.llm_client.temperature = original_temperature

    @staticmethod
    def _critic_qtypes(critique: str, qtypes: List[str]) -> Set[str]:
        return {qtype for qtype in qtypes if re.search(rf"{re.escape(qtype)}[型题]?", critique)}

    @staticmethod
    def _critique_for_qtype(critique: str, qtype: str) -> str:
        lines = [line.strip() for line in critique.splitlines() if qtype in line and line.strip()]
        return "\n".join(lines) or critique

    def _repair_qtype(
        self,
        qtype: str,
        structured_case: dict,
        soap_content: str,
        rag_context: str,
        previous_block: str,
        critique: str,
    ) -> str:
        """选择性修复白名单：禁止原文和 reasoning 进入该调用。"""
        payload = {
            "requested_type": qtype,
            "requirements": QTYPE_REQUIREMENTS[qtype],
            "structured_case": structured_case,
            "soap_content": soap_content,
            "rag_context": rag_context,
            "few_shot_example": self._load_few_shot(qtype),
            "previous_output": previous_block,
            "critique": critique,
        }
        return self._invoke("question/executor_repair_single.md", payload)

    def _reflect_stage(
        self,
        output: str,
        qtypes: List[str],
        structured_case: dict,
        soap_content: str,
        rag_context: str,
        label: str,
    ) -> str:
        if not self.enable_reflection:
            return output
        rule_issues = self._rule_check(output, qtypes)
        critique = self._invoke_critic(output, qtypes)
        if not rule_issues and critique.strip().upper() == "OK":
            return output

        problematic = {item["qtype"] for item in rule_issues} | self._critic_qtypes(critique, qtypes)
        if not problematic:
            problematic = set(qtypes)
        parts: List[str] = []
        for qtype in qtypes:
            block = self._extract_qtype_block(output, qtype)
            if qtype not in problematic:
                parts.append(block)
                continue
            rule_text = "\n".join(item["text"] for item in rule_issues if item["qtype"] == qtype)
            repair_critique = self._critique_for_qtype(critique, qtype)
            if rule_text:
                repair_critique = rule_text + ("\n" + repair_critique if repair_critique else "")
            parts.append(self._repair_qtype(qtype, structured_case, soap_content, rag_context, block, repair_critique))
        repaired = "\n---\n".join(part for part in parts if part)
        remaining = self._rule_check(repaired, qtypes)
        if remaining:
            print(f"[REFLECT] {label} repaired with remaining rule issues: {[item['text'] for item in remaining]}")
        return repaired

    def _initial_payload(
        self,
        structured_case: dict,
        case_text: str,
        case_flags: dict,
        soap_content: str,
        reasoning_content: str,
        rag_context: str,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "structured_case": structured_case,
            "case_text": case_text,
            "case_flags": case_flags,
            "soap_content": soap_content,
            "reasoning_content": reasoning_content,
        }
        if rag_context:
            payload["rag_context"] = rag_context
        return payload

    def _generate_batch(self, payload: Dict[str, Any]) -> str:
        request = dict(payload)
        request["requested_types"] = BATCH_TYPES
        request["few_shot_examples"] = {qtype: self._load_few_shot(qtype) for qtype in BATCH_TYPES}
        return self._invoke("question/executor_batch_a123.md", request)

    def _generate_single(self, qtype: str, payload: Dict[str, Any]) -> str:
        request = dict(payload)
        request["requested_type"] = qtype
        request["few_shot_example"] = self._load_few_shot(qtype)
        return self._invoke("question/executor.md", request)

    def execute(
        self,
        structured_case: dict,
        reasoning_content: str = "",
        soap_content: str = "",
        case_text: str = "",
        case_flags: dict | None = None,
    ) -> str:
        rag_context = self._retrieve_context(structured_case, raw_text=case_text).get("context", "")
        payload = self._initial_payload(structured_case, case_text, case_flags or {}, soap_content, reasoning_content, rag_context)
        batch = self._generate_batch(payload)
        a4 = self._generate_single("A4", payload)
        b1 = self._generate_single("B1", payload)
        batch = self._reflect_stage(batch, BATCH_TYPES, structured_case, soap_content, rag_context, "A1-A3")
        a4 = self._reflect_stage(a4, ["A4"], structured_case, soap_content, rag_context, "A4")
        b1 = self._reflect_stage(b1, ["B1"], structured_case, soap_content, rag_context, "B1")
        return "# 临床执业医师考试试题\n\n" + "\n---\n".join([batch, a4, b1])
