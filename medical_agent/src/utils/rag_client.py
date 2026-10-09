"""Lightweight RAG client for medical_text — reuses medical_agent's Chroma DB."""
from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional


_MA_ROOT_ENV = os.getenv("MEDICAL_RAG_RUNTIME_ROOT")
_MA_ROOT: Optional[Path] = (
    Path(_MA_ROOT_ENV).expanduser().resolve() if _MA_ROOT_ENV else None
)


def _load_module(module_path: Path, fullname: str):
    spec = importlib.util.spec_from_file_location(fullname, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_path} as {fullname}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_package(fullname: str, search_locations: List[str]):
    if fullname not in sys.modules:
        pkg = types.ModuleType(fullname)
        pkg.__path__ = search_locations
        sys.modules[fullname] = pkg


# ---------------------------------------------------------------------------
# Meta-path finder: intercepts imports under  src.rag / src.utils / src.models
# and transparently redirects them to the medical_agent equivalents under
# _ma_src so that medical_text's own  src.*  imports remain untouched.
# ---------------------------------------------------------------------------
class _MedicalAgentAliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    _ALIASED_PREFIXES = ("src.rag.", "src.utils.", "src.models.")
    _ALIASED_PACKAGES = {"src.rag", "src.utils", "src.models"}

    def find_spec(self, fullname: str, path: Any, target: Any = None):
        if fullname.startswith(self._ALIASED_PREFIXES) or fullname in self._ALIASED_PACKAGES:
            alias = fullname.replace("src.", "_ma_src.", 1)
            if alias in sys.modules:
                return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        alias = spec.name.replace("src.", "_ma_src.", 1)
        return sys.modules.get(alias)

    def exec_module(self, module):
        pass  # already loaded


# Install the finder before any medical_agent module is loaded.
if _MedicalAgentAliasFinder not in sys.meta_path:
    sys.meta_path.insert(0, _MedicalAgentAliasFinder())


_RAGRetriever: Any = None
_IMPORT_ERROR: Optional[str] = None

try:
    if _MA_ROOT is None:
        raise RuntimeError("MEDICAL_RAG_RUNTIME_ROOT 未设置；请指向只读 RAG 运行时目录")
    _ma_src = _MA_ROOT / "src"

    # Build empty package hierarchy for medical_agent
    _ensure_package("_ma_src", [str(_ma_src)])
    for sub in ("utils", "rag", "models"):
        _ensure_package(f"_ma_src.{sub}", [str(_ma_src / sub)])

    # Load leaf modules (skip __init__.py to avoid circular imports)
    for py_file in sorted((_ma_src / "utils").glob("*.py"), key=lambda p: p.name):
        if py_file.name == "__init__.py":
            continue
        _load_module(py_file, f"_ma_src.utils.{py_file.stem}")

    for py_file in sorted((_ma_src / "models").glob("*.py"), key=lambda p: p.name):
        if py_file.name == "__init__.py":
            continue
        _load_module(py_file, f"_ma_src.models.{py_file.stem}")

    for py_file in sorted((_ma_src / "rag").glob("*.py"), key=lambda p: p.name):
        if py_file.name == "__init__.py":
            continue
        _load_module(py_file, f"_ma_src.rag.{py_file.stem}")

    _RAGRetriever = sys.modules["_ma_src.rag.retriever"].RAGRetriever
except Exception as exc:
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


class RAGClient:
    """Thin wrapper around medical_agent's RAGRetriever."""

    def __init__(
        self,
        ma_root: Optional[Path] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled and _RAGRetriever is not None
        self._error: Optional[str] = None
        self._retriever: Any = None

        if not self.enabled:
            self._error = _IMPORT_ERROR or "RAGRetriever unavailable"
            if enabled:
                raise RuntimeError(f"RAG 初始化失败: {self._error}")
            return

        root = (ma_root or _MA_ROOT).resolve()
        rag_config = root / "config" / "rag_config.yaml"
        llm_config = root / "config" / "llm_config.yaml"

        if not rag_config.exists():
            self.enabled = False
            self._error = f"rag_config not found: {rag_config}"
            raise RuntimeError(f"RAG 初始化失败: {self._error}")

        try:
            self._retriever = _RAGRetriever(
                project_root=root,
                rag_config_path=rag_config,
                llm_config_path=llm_config,
                enabled_override=True,
            )
        except Exception as exc:
            self.enabled = False
            self._error = str(exc)
            raise RuntimeError(f"RAG 初始化失败: {self._error}") from exc

    def ensure_ready(self) -> Dict[str, Any]:
        if not self.enabled or self._retriever is None:
            raise RuntimeError(f"RAG 不可用: {self._error or 'retriever unavailable'}")
        return self._retriever.ensure_ready()

    def retrieve(
        self,
        query_plan: List[Dict[str, str]],
        agent_name: str = "",
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self.enabled or self._retriever is None:
            raise RuntimeError(f"RAG 不可用: {self._error or 'retriever unavailable'}")
        return self._retriever.retrieve(
            query_plan=query_plan,
            agent_name=agent_name,
            top_k=top_k,
        )

    @staticmethod
    def build_query_plan(
        task_name: str,
        structured_case: dict,
        raw_text: str = "",
    ) -> List[Dict[str, str]]:
        def _clean(value: Any) -> str:
            if value is None:
                return ""
            text = str(value).strip()
            return "" if text.lower() == "none" else text

        basic_info = structured_case.get("basic_info", {}) or {}
        department = _clean(basic_info.get("department"))
        chief_complaint = _clean(structured_case.get("chief_complaint"))
        diagnosis = _clean(structured_case.get("diagnosis"))
        history = _clean(structured_case.get("history"))
        physical_exam = _clean(structured_case.get("physical_exam"))
        auxiliary_exam = _clean(structured_case.get("auxiliary_exam"))
        treatment = _clean(structured_case.get("treatment"))

        def _raw_excerpt(labels: List[str], max_chars: int = 300) -> str:
            if not raw_text:
                return ""
            for label in labels:
                idx = raw_text.find(label)
                if idx >= 0:
                    return raw_text[idx: idx + max_chars].replace("\n", " ").strip()
            return raw_text[:max_chars].replace("\n", " ").strip()

        raw_core = _raw_excerpt(["诊断", "主诉", "现病史", "病例"], 260)
        raw_clinical = _raw_excerpt(["现病史", "体格检查", "检查", "辅助检查"], 360)
        raw_management = _raw_excerpt(["治疗", "医嘱", "手术", "出院"], 320)

        # Core keywords extracted from diagnosis / chief complaint (highest priority)
        core_terms = " ".join(p for p in [department, chief_complaint, diagnosis] if p) or raw_core

        # Supplemental clinical context (truncated to avoid noise)
        def _clip(text: str, max_chars: int = 120) -> str:
            if not text:
                return ""
            # Prefer first sentence / line as summary
            first_line = text.split("\n")[0]
            clipped = first_line[:max_chars]
            if len(first_line) > max_chars:
                clipped = clipped[: max_chars - 3] + "..."
            return clipped

        history_clip = _clip(history, 100)
        physical_clip = _clip(physical_exam, 100)
        auxiliary_clip = _clip(auxiliary_exam, 120)
        treatment_clip = _clip(treatment, 100)

        # Build task-specific query aspects
        queries: List[Dict[str, str]] = []

        # Aspect 1: core identity of the case
        if core_terms:
            queries.append({"name": "core", "query": core_terms})

        # Aspect 2: clinical presentation & exam (key for reasoning)
        clinical_parts = [p for p in [history_clip, physical_clip, auxiliary_clip] if p] or ([raw_clinical] if raw_clinical else [])
        if clinical_parts:
            queries.append({"name": "clinical", "query": " ".join(clinical_parts)})

        # Aspect 3: treatment / management (key for question generation)
        treatment_parts = [p for p in [diagnosis, treatment_clip, auxiliary_clip] if p] or ([raw_management] if raw_management else [])
        if treatment_parts:
            queries.append({"name": "management", "query": " ".join(treatment_parts)})

        if not queries:
            return []
        return queries
