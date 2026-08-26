from typing import TypedDict, Optional, List, Dict, Annotated
import operator

class AgentState(TypedDict, total=False):
    # Core User Inputs
    user_prompt: str
    file_path: Optional[str]

    # Orchestrator & Multi-Step Planning
    execution_plan: List[str]          # e.g., ['doc_read', 'rag_search', 'coding_agent', 'doc_write']
    current_step_index: int            # Index of current step in execution_plan
    step_models: Dict[str, str]        # Auto-selected model per step

    # Inter-Agent Data Context
    extracted_text: Optional[str]      # From Member 3 (Doc Read / OCR / Vision)
    retrieved_context: Optional[str]   # From Member 4 (RAG Knowledge Base)
    sop_citations: List[str]

    # Member 5 Coding Pipeline
    generated_code: Optional[str]
    execution_output: Optional[str]
    execution_error: Optional[str]
    test_passed: bool
    review_feedback: Optional[str]
    review_passed: bool
    retry_count: int

    # Member 6 Deliverables & Quality Gates
    validation_status: str             # "passed" | "failed"
    guardrails_passed: bool
    artifact_path: Optional[str]
    human_approved: bool

    # Error & Audit
    error_message: Optional[str]
    execution_history: Annotated[List[Dict[str, str]], operator.add]