from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator

class AgentState(TypedDict, total=False):
    # Member 1: Watchman
    file_path: Optional[str]
    file_type: Optional[str]
    task_type: Optional[str]
    user_prompt: str

    # Member 2: Orchestrator / Planner
    execution_plan: List[str]
    current_step_index: int
    # Which local model was auto-selected for each step, e.g. {"coding_agent": "qwen2.5-coder:7b"}
    step_models: Dict[str, str]
    # Reducer preserves full audit trail across all nodes rather than overwriting
    execution_history: Annotated[List[Dict[str, Any]], operator.add]
    model_name: str
    error_message: Optional[str]

    # Member 3: Document Reading & Multimodal
    extracted_text: Optional[str]
    extracted_tables: Optional[List[Dict[str, Any]]]
    extracted_images_summary: Optional[str]
    is_scanned: bool

    # Member 4: RAG Knowledge Base
    retrieved_sop_context: Optional[str]
    # Reducer accumulates citations from multiple queries
    sop_citations: Annotated[List[str], operator.add]

    # Member 5: Coding & Testing Sandbox
    generated_code: Optional[str]
    execution_output: Optional[str]
    execution_error: Optional[str]
    test_passed: bool
    review_feedback: Optional[str]
    retry_count: int

    # Member 6: Output, Validation & Guardrails
    generated_artifact_path: Optional[str]
    validation_status: str          # "passed", "failed"
    validation_errors: List[str]
    guardrails_passed: bool
    human_approved: Optional[bool]