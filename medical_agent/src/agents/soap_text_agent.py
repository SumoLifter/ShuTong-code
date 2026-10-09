"""Pure-text SOAP agent — outputs Markdown SOAP note instead of JSON."""
from __future__ import annotations

from src.agents.base_agent import BaseAgent


class SOAPTextAgent(BaseAgent):
    def __init__(self, provider: str, model: str, temperature: float = 0.2, use_rag: bool = True):
        super().__init__("soap", provider, model, temperature, use_rag=use_rag)

    def execute(self, structured_case: dict, case_text: str = "", case_flags: dict | None = None) -> str:
        rag_result = self._retrieve_context(structured_case, raw_text=case_text)
        payload = {"structured_case": structured_case, "case_text": case_text, "case_flags": case_flags or {}}
        if rag_result.get("context"):
            payload["rag_context"] = rag_result["context"]
        return self._invoke("soap/executor.md", payload)
