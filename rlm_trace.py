# rlm_trace.py
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional


def env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class TraceConfig:
    verbose: bool
    trace: bool
    trace_path: str

    @staticmethod
    def from_env() -> "TraceConfig":
        return TraceConfig(
            verbose=env_bool("RLM_VERBOSE", False),
            trace=env_bool("RLM_TRACE", False),
            trace_path=os.getenv("RLM_TRACE_PATH", "rlm_trace_tree.json"),
        )


class TraceTree:
    """
    Tree-structured trace.
    nodes[node_id] = { id, parent_id, type, label, ts_start_ms, ts_end_ms, data, children[] }
    """

    def __init__(self, cfg: Optional[TraceConfig] = None):
        self.cfg = cfg or TraceConfig.from_env()
        self.root_id: str = self._new_id()
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.nodes[self.root_id] = {
            "id": self.root_id,
            "parent_id": None,
            "type": "run",
            "label": "rlm.run",
            "ts_start_ms": now_ms(),
            "ts_end_ms": None,
            "data": {},
            "children": [],
        }

    def _new_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def _vlog(self, msg: str) -> None:
        if self.cfg.verbose:
            print(msg)

    def node(
        self,
        *,
        parent_id: Optional[str],
        node_type: str,
        label: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> str:
        if parent_id is None:
            parent_id = self.root_id
        if parent_id not in self.nodes:
            raise KeyError(f"parent_id not found: {parent_id}")

        node_id = self._new_id()
        self.nodes[node_id] = {
            "id": node_id,
            "parent_id": parent_id,
            "type": node_type,
            "label": label,
            "ts_start_ms": now_ms(),
            "ts_end_ms": None,
            "data": data or {},
            "children": [],
        }
        self.nodes[parent_id]["children"].append(node_id)
        self._vlog(f"[{node_type}] {label} id={node_id} parent={parent_id}")
        return node_id

    def update(self, node_id: str, **data_updates: Any) -> None:
        if node_id not in self.nodes:
            raise KeyError(f"node_id not found: {node_id}")
        self.nodes[node_id]["data"].update(data_updates)
        self._vlog(f"[update] id={node_id} keys={', '.join(data_updates.keys())}")

    def end(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise KeyError(f"node_id not found: {node_id}")
        self.nodes[node_id]["ts_end_ms"] = now_ms()
        self._vlog(f"[end] id={node_id} label={self.nodes[node_id]['label']}")

    def flush(self, run_meta: Optional[Dict[str, Any]] = None) -> None:
        if self.nodes[self.root_id]["ts_end_ms"] is None:
            self.nodes[self.root_id]["ts_end_ms"] = now_ms()

        if not self.cfg.trace:
            return

        payload = {
            "run_meta": run_meta or {},
            "root_id": self.root_id,
            "nodes": self.nodes,
        }
        with open(self.cfg.trace_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        self._vlog(f"[trace] wrote {self.cfg.trace_path}")
