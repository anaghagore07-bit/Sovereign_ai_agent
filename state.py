from typing import TypedDict, Optional, Literal, Dict, Any, List

class AgentState(TypedDict, total=False):
    # Core User Input & Planning
    user_prompt: str
    attached_files: List[str]
    plan: List[str]
    current_step_index: int
    selected_route: Optional[str]
    
    # Member 3: Document Reading / OCR / Vision
    extracted_data: Optional[str]
    extracted_doc_data: Optional[Dict[str, Any]]
    
    # Member 4: RAG & SOP Knowledge
    sop_rules: Optional[str]
    retrieved_sop_context: Optional[str]
    
    # Member 5: Coding & Sandbox Verification
    generated_code: Optional[str]
    code_results: Optional[str]
    test_status: Literal["pass", "fail", "not_run"]
    retry_count: int
    error_log: Optional[str]
    
    # Member 6: Output & Artifact Generation
    generated_artifact: Optional[str]
    final_output: Optional[str]
    artifact_path: Optional[str]