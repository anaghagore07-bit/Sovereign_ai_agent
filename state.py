"""
Canonical shared state for the Sovereign AI multi-agent system.

This is the single source of truth for AgentState. Every node file
(watchman, orchestrator/graph, doc_read, rag, coding, doc_write, ...)
should `from state import AgentState` instead of redefining its own
version — two divergent AgentState definitions is what caused the
schema mismatch between graph.py and this file in the first draft.
"""
from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator


class AgentState(TypedDict, total=False):
    # ---- Member 1: Watchman / file intake ----
    file_path: Optional[str]
    file_type: Optional[str]
    task_type: Optional[str]        # Member 1's coarse file/task classification
    user_prompt: str

    # ---- Member 2: Orchestrator / Planner / Model routing ----
    route: Optional[str]            # single-category route: "coding" | "doc_read" | "doc_write" | "clarification"
    execution_plan: List[str]       # ordered high-level stages, e.g. ["doc_read", "coding_agent", "doc_write"]
    current_step_index: int
    step_models: Dict[str, str]     # local model auto-selected per node/engine name
    model_name: str
    error_message: Optional[str]
    # Reducer preserves the full audit trail across all nodes rather than overwriting it
    execution_history: Annotated[List[Dict[str, Any]], operator.add]

    # ---- Member 3: Document Reading & Multimodal ----
    sub_read_type: Optional[str]        # ingestion engine currently active: "ocr" | "rag" | "vision"
    sub_read_queue: List[str]           # remaining ingestion engines still queued for this request
    extracted_text: Optional[str]
    extracted_tables: Optional[List[Dict[str, Any]]]
    extracted_images_summary: Optional[str]
    is_scanned: bool

    # ---- Member 4: RAG Knowledge Base ----
    retrieved_sop_context: Optional[str]
    retrieved_context: Optional[str]        # alias some nodes read; kept for compatibility
    sop_citations: Annotated[List[str], operator.add]

    # ---- Member 5: Coding & Testing Sandbox ----
    generated_code: Optional[str]
    execution_output: Optional[str]
    execution_error: Optional[str]
    test_passed: bool
    review_passed: bool
    review_feedback: Optional[str]
    retry_count: int

    # ---- Member 6: Output, Validation & Guardrails ----
    generated_artifact_path: Optional[str]
    validation_status: str          # "passed" | "failed"
    validation_errors: List[str]
    guardrails_passed: bool
    human_approved: Optional[bool]