import re
import asyncio
from typing import Optional, List, Dict, Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from state import AgentState
from planner import (
    level_2_llm_router,
    build_execution_plan,
    build_step_models,
    get_model_for_step,
    ROUTER_MODEL,
)

MAX_RETRIES = 3

# ==========================================
# 1. TIER 1: REGEX ROUTER
# ==========================================
CODING_PATTERN = re.compile(r"\b(calculate|compute|script|code|verify|equation|math)\b", re.I)
DOC_READ_PATTERN = re.compile(r"\b(read|ocr|scan|vision|rag|sop|clause|manual|extract|diagram|drawing)\b", re.I)
DOC_WRITE_PATTERN = re.compile(r"\b(write|draft|report|note|summary|approval)\b", re.I)
VALIDATE_PATTERN = re.compile(r"\b(validate|check|audit|guardrail|inspect)\b", re.I)
NONSENSE_PATTERN = re.compile(r"^[a-z]{1,6}$", re.I)

def level_1_regex_router(prompt: str, file_path: Optional[str]) -> Optional[str]:
    prompt_str = (prompt or "").strip()
    has_file = bool(file_path and file_path.strip())

    if not prompt_str or (NONSENSE_PATTERN.match(prompt_str) and not has_file):
        return "clarification"

    if has_file or DOC_READ_PATTERN.search(prompt_str):
        return "doc_read"
    if CODING_PATTERN.search(prompt_str):
        return "coding"
    if DOC_WRITE_PATTERN.search(prompt_str):
        return "doc_write"
    if VALIDATE_PATTERN.search(prompt_str):
        return "output_validator"

    return None

# ==========================================
# 2. WATCHMAN & ORCHESTRATOR NODES
# ==========================================
async def watchman_node(state: AgentState) -> Dict[str, Any]:
    file_path = state.get("file_path")
    status = f"Ingested file: {file_path}" if file_path else "Direct prompt intake"
    print(f"\n[Watchman] Intake: {status}")
    return {"execution_history": [{"node": "watchman", "status": status}]}

async def orchestrator_node(state: AgentState) -> Dict[str, Any]:
    prompt = state.get("user_prompt", "")
    file_path = state.get("file_path")
    has_file = bool(file_path and file_path.strip())

    route = level_1_regex_router(prompt, file_path)
    tier = "1 (regex)"
    if route is None:
        route = level_2_llm_router(prompt, has_file)
        tier = "2 (LLM)"

    plan = build_execution_plan(route, prompt)
    step_models = build_step_models(plan)
    first_model = step_models.get(plan[0], ROUTER_MODEL) if plan else ROUTER_MODEL

    print(f"[Orchestrator] Tier {tier} -> selected route: '{route}', plan: {plan}")

    return {
        "route": route,
        "execution_plan": plan,
        "current_step_index": 0,
        "step_models": step_models,
        "model_name": first_model,
        "retry_count": 0,
        "execution_history": [{
            "node": "orchestrator",
            "status": f"routed_to_{route}",
            "tier": tier,
            "plan": plan,
        }],
    }

def route_orchestrator(state: AgentState) -> str:
    route = state.get("route") or "clarification"
    # Match diagram branch names
    mapping = {
        "coding": "coding_agent",
        "doc_read": "doc_read",
        "doc_write": "doc_write",
        "output_validator": "output_validator",
        "clarification": "clarification_node",
    }
    return mapping.get(route, "clarification_node")

async def clarification_node(state: AgentState) -> Dict[str, Any]:
    msg = "I couldn't map that request to a supported task. Please clarify your prompt or provide a file."
    print(f"[Clarification] {msg}")
    return {
        "error_message": msg,
        "validation_status": "failed",
        "guardrails_passed": False,
        "human_approved": False,
        "execution_history": [{"node": "clarification_node", "status": "needs_clarification"}],
    }

# ==========================================
# 3. MEMBER 3: DOC READ
# ==========================================
async def doc_read_node(state: AgentState) -> Dict[str, Any]:
    print("--> [Doc Read] Parsing document (OCR / Vision / RAG)...")
    await asyncio.sleep(0.05)
    return {
        "extracted_text": "[Extracted technical specifications from document]",
        "execution_history": [{"node": "doc_read", "status": "completed"}],
    }

# ==========================================
# 4. MEMBER 5: CODING PIPELINE
# ==========================================
async def coding_agent_node(state: AgentState) -> Dict[str, Any]:
    model = state.get("step_models", {}).get("coding_agent", get_model_for_step("coding_agent"))
    print(f"--> [Coding Agent] Generating code with model='{model}'...")
    await asyncio.sleep(0.05)
    return {
        "generated_code": "thickness = 12.4\nresult = thickness / 10.0\nprint(f'SAFETY_FACTOR:{result:.2f}')",
        "model_name": model,
        "execution_history": [{"node": "coding_agent", "status": "code_generated"}],
    }

async def code_test_agent_node(state: AgentState) -> Dict[str, Any]:
    print("--> [Code Test Agent] Executing script in sandbox...")
    await asyncio.sleep(0.05)
    passed = True  # Sandbox status placeholder

    update: Dict[str, Any] = {
        "execution_output": "SAFETY_FACTOR:1.24",
        "test_passed": passed,
        "execution_history": [{"node": "code_test_agent", "status": "passed" if passed else "failed"}],
    }
    if not passed:
        update["retry_count"] = state.get("retry_count", 0) + 1
        update["execution_error"] = "Sandbox assertion failure"
    return update

def route_code_test(state: AgentState) -> str:
    if state.get("test_passed"):
        return "code_review_agent"
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "coding_agent"
    print("[Coding Loop] Max retries exceeded during sandbox test — escalating to artifact generator.")
    return "artifact_generator"

async def code_review_agent_node(state: AgentState) -> Dict[str, Any]:
    print("--> [Code Review Agent] Performing quality and security review...")
    await asyncio.sleep(0.05)
    passed = bool(state.get("test_passed"))

    update: Dict[str, Any] = {
        "review_passed": passed,
        "review_feedback": "Code conforms to standard specifications" if passed else "Quality check failed",
        "execution_history": [{"node": "code_review_agent", "status": "approved" if passed else "rejected"}],
    }
    if not passed:
        update["retry_count"] = state.get("retry_count", 0) + 1
    return update

def route_code_review(state: AgentState) -> str:
    if state.get("review_passed"):
        return "artifact_generator"
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "coding_agent"
    print("[Coding Loop] Max retries exceeded during review — escalating to artifact generator.")
    return "artifact_generator"

# ==========================================
# 5. MEMBER 6: WRITING, VALIDATION, GUARDRAILS & HITL
# ==========================================
async def doc_write_node(state: AgentState) -> Dict[str, Any]:
    model = state.get("step_models", {}).get("doc_write", get_model_for_step("doc_write"))
    print(f"--> [Doc Write] Generating draft report with model='{model}'...")
    await asyncio.sleep(0.05)
    return {
        "model_name": model,
        "execution_history": [{"node": "doc_write", "status": "draft_created"}],
    }

async def output_validator_node(state: AgentState) -> Dict[str, Any]:
    print("--> [Output Validator] Verifying schemas and field integrity...")
    return {
        "validation_status": "passed",
        "validation_errors": [],
        "execution_history": [{"node": "output_validator", "status": "passed"}],
    }

async def guardrails_check_node(state: AgentState) -> Dict[str, Any]:
    print("--> [Guardrails Check] Checking compliance and scanning for data leaks...")
    return {
        "guardrails_passed": True,
        "execution_history": [{"node": "guardrails_check", "status": "passed"}],
    }

async def artifact_generator_node(state: AgentState) -> Dict[str, Any]:
    print("--> [Artifact Generator] Building final deliverable files...")
    return {
        "generated_artifact_path": "exports/Engineering_Compliance_Report.docx",
        "execution_history": [{"node": "artifact_generator", "status": "created"}],
    }

async def human_approval_node(state: AgentState) -> Dict[str, Any]:
    artifact = state.get("generated_artifact_path", "No artifact")
    decision = interrupt({
        "question": "Approve this deliverable?",
        "artifact": artifact,
        "validation": state.get("validation_status"),
        "guardrails": state.get("guardrails_passed")
    })
    approved = bool(decision) if isinstance(decision, bool) else str(decision).lower() in ("yes", "true", "approve", "y")
    print(f"--> [Human Approval] Decision={decision} -> Approved={approved}")
    return {
        "human_approved": approved,
        "execution_history": [{"node": "human_approval", "approved": approved}],
    }

# ==========================================
# 6. GRAPH ASSEMBLY
# ==========================================
def build_graph():
    graph = StateGraph(AgentState)

    # Register Nodes
    graph.add_node("watchman", watchman_node)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("clarification_node", clarification_node)
    graph.add_node("coding_agent", coding_agent_node)
    graph.add_node("code_test_agent", code_test_agent_node)
    graph.add_node("code_review_agent", code_review_agent_node)
    graph.add_node("doc_read", doc_read_node)
    graph.add_node("doc_write", doc_write_node)
    graph.add_node("output_validator", output_validator_node)
    graph.add_node("guardrails_check", guardrails_check_node)
    graph.add_node("artifact_generator", artifact_generator_node)
    graph.add_node("human_approval", human_approval_node)

    # Start -> Watchman -> Orchestrator
    graph.add_edge(START, "watchman")
    graph.add_edge("watchman", "orchestrator")

    # Orchestrator Multi-Branching (Matching Diagram Exactly)
    graph.add_conditional_edges(
        "orchestrator",
        route_orchestrator,
        {
            "coding_agent": "coding_agent",
            "doc_read": "doc_read",
            "doc_write": "doc_write",
            "output_validator": "output_validator",
            "clarification_node": "clarification_node",
        }
    )

    # Clarification Fast-Exit
    graph.add_edge("clarification_node", END)

    # Doc Read -> Doc Write Path
    graph.add_edge("doc_read", "doc_write")
    graph.add_edge("doc_write", "output_validator")
    graph.add_edge("output_validator", "guardrails_check")
    graph.add_edge("guardrails_check", "artifact_generator")

    # Coding Loop & Retry Paths
    graph.add_edge("coding_agent", "code_test_agent")
    graph.add_conditional_edges(
        "code_test_agent",
        route_code_test,
        {
            "coding_agent": "coding_agent",
            "code_review_agent": "code_review_agent",
            "artifact_generator": "artifact_generator",
        }
    )
    graph.add_conditional_edges(
        "code_review_agent",
        route_code_review,
        {
            "coding_agent": "coding_agent",
            "artifact_generator": "artifact_generator",
        }
    )

    # Deliverable & Human Approval Gate
    graph.add_edge("artifact_generator", "human_approval")
    graph.add_edge("human_approval", END)

    return graph.compile(checkpointer=MemorySaver())

# ==========================================
# 7. EXECUTION HARNESS
# ==========================================
async def main():
    app = build_graph()
    config = {"configurable": {"thread_id": "session_diagram_v2"}}

    initial_state: AgentState = {
        "user_prompt": "Read the attached inspection document and generate the approval note.",
        "file_path": "uploads/inspection_report.pdf",
        "retry_count": 0,
        "current_step_index": 0,
        "execution_history": [],
        "sop_citations": [],
    }

    print("\n=== STARTING UPDATED PIPELINE WORKFLOW ===")
    result = await app.ainvoke(initial_state, config=config)

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print(f"\n[HUMAN APPROVAL REQUIRED] {payload['question']}")
        print(f"Artifact: {payload['artifact']}")
        print(f"Validation: {payload['validation']} | Guardrails: {payload['guardrails']}")
        
        while True:
            choice = input("Approve deliverable? (yes/no): ").strip().lower()
            if choice in ("yes", "y", "no", "n"):
                break
            print("Please type 'yes' or 'no'.")

        result = await app.ainvoke(Command(resume=choice), config=config)

    print("\n=== FINAL PIPELINE SUMMARY ===")
    print("Route Chosen:", result.get("route"))
    print("Artifact Created:", result.get("generated_artifact_path"))
    print("Validation Status:", result.get("validation_status"))
    print("Guardrails Passed:", result.get("guardrails_passed"))
    print("Human Approved:", result.get("human_approved"))
    print("Total History Steps:", len(result.get("execution_history", [])))

if __name__ == "__main__":
    asyncio.run(main())