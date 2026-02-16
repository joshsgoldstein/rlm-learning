from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Protocol

from rlm_core import AnswerResult, ChatMessage, Config


@dataclass
class ApproachRun:
    answer: AnswerResult
    metadata: Dict[str, object] = field(default_factory=dict)


class Approach(Protocol):
    id: str
    label: str
    style: str

    def run(
        self,
        *,
        doc_map: Dict[str, str],
        question: str,
        history: List[ChatMessage],
        cfg: Config,
        on_event: Optional[Callable] = None,
    ) -> ApproachRun:
        ...


class BaseApproach(ABC):
    id: str
    label: str
    style: str = "dim"

    @abstractmethod
    def run(
        self,
        *,
        doc_map: Dict[str, str],
        question: str,
        history: List[ChatMessage],
        cfg: Config,
        on_event: Optional[Callable] = None,
    ) -> ApproachRun:
        raise NotImplementedError


class TraditionalApproach(BaseApproach):
    id = "traditional"
    label = "Traditional"
    style = "dim red"

    def run(
        self,
        *,
        doc_map: Dict[str, str],
        question: str,
        history: List[ChatMessage],
        cfg: Config,
        on_event: Optional[Callable] = None,
    ) -> ApproachRun:
        from .traditional import answer_traditional

        result = answer_traditional(doc_map=doc_map, question=question, cfg=cfg, on_event=on_event)
        return ApproachRun(answer=result, metadata={})


class RagApproach(BaseApproach):
    id = "rag"
    label = "RAG"
    style = "magenta"

    def run(
        self,
        *,
        doc_map: Dict[str, str],
        question: str,
        history: List[ChatMessage],
        cfg: Config,
        on_event: Optional[Callable] = None,
    ) -> ApproachRun:
        from .rag import answer_rag

        result, stats = answer_rag(doc_map=doc_map, question=question, cfg=cfg, on_event=on_event)
        return ApproachRun(answer=result, metadata={"rag_stats": asdict(stats)})


class RLMApproach(BaseApproach):
    id = "rlm"
    label = "RLM"
    style = "green"

    def run(
        self,
        *,
        doc_map: Dict[str, str],
        question: str,
        history: List[ChatMessage],
        cfg: Config,
        on_event: Optional[Callable] = None,
    ) -> ApproachRun:
        from .rlm import answer_question

        result = answer_question(
            doc_map=doc_map,
            question=question,
            history=history,
            cfg=cfg,
            on_event=on_event,
        )
        return ApproachRun(answer=result, metadata={})


_APPROACH_REGISTRY: Dict[str, BaseApproach] = {
    "rlm": RLMApproach(),
    "traditional": TraditionalApproach(),
    "rag": RagApproach(),
}


def register_approach(approach: BaseApproach) -> None:
    """Register a custom approach implementation by id."""
    _APPROACH_REGISTRY[approach.id] = approach


def get_approach(approach_id: str) -> BaseApproach:
    if approach_id not in _APPROACH_REGISTRY:
        raise KeyError(f"Unknown approach '{approach_id}'. Registered: {', '.join(sorted(_APPROACH_REGISTRY.keys()))}")
    return _APPROACH_REGISTRY[approach_id]


def list_approaches() -> List[str]:
    return list(_APPROACH_REGISTRY.keys())
