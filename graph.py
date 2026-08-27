import re
import asyncio
from typing import Optional, List, Dict, Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from state import AgentState
from planner import (
    level_2_llm_router,
    build_execution_plan,
    build_step_models,
    determine_sub_engines,
    get_model_for_step,
    ROUTER_MODEL,
)

MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Tier 1: fast regex router. Returns None (not a guess) when ambiguous, so
# the caller knows to fall back to the Tier-2 LLM router instead of silently
# mis-routing.
# ---------------------------------------------------------------------------
CODING_PATTERN = re.compile(r"\b(calculate|compute|script|code|verify|equation|math)\b", re.I)
DOC_READ_PATTERN = re.compile(r"\b(read|ocr|scan|vision|rag|sop|clause|manual|extract|diagram|drawing)\b", re.I)
DOC_WRITE_PATTERN = re.compile(r"\b(write|draft|report|note|summary|approval)\b", re.I)
NONSENSE_PATTERN = re.compile(r"^[a-z]{1,6}$", re.I)  # e.g. "asdfg", "qwerty"


def level_1_regex_router(prompt: str, file_path: Optional[str]) -> Optional[str]:
    prompt = (prompt or "").strip()
    has_file = bool(file_path and file_path.strip())

    if not prompt or (NONSENSE_PATTERN.match(prompt) and not has_file):
        return "clarification"

    # A file always implies at least an ingestion step; whether coding/
    # doc_write also runs is decided later by build_execution_plan, which
    # reads the same prompt for coding-intent keywords. This keeps "route a
    # file to doc_read" and "does this task also need coding" as two
    # separate, non-conflicting decisions instead of one router trying to
    # do both.
    if has_file:
        return "doc_read"

    if DOC_READ_PATTERN.search(prompt):
        return "doc_read"
    if CODING_PATTERN.search(prompt):
        return "coding"
    if DOC_WRITE_PATTERN.search(prompt):
        return "doc_write"

    return None  # ambiguous -> Tier 2


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
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

    print(f"[Orchestrator] Tier {tier} -> route='{route}', plan={plan}")

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
    # orchestrator_node always resolves a concrete route (Tier 2 itself
    # defaults to "clarification" on any failure), so this default is a
    # last-resort safety net rather than the normal path.
    return state.get("route") or "clarification"


async def clarification_node(state: AgentState) -> Dict[str, Any]:
    msg = "I couldn't map that request to a supported task. Could you rephrase or attach a file?"
    print(f"[Clarification] {msg}")
    return {
        "error_message": msg,
        "validation_status": "failed",
        "guardrails_passed": False,
        "human_approved": None,
        "execution_history": [{"node": "clarification", "status": "needs_clarification"}],
    }


# ---------------------------------------------------------------------------
# Member 3: doc_read + multi-engine ingestion loop
# ---------------------------------------------------------------------------
async def doc_read_node(state: AgentState) -> Dict[str, Any]:
    prompt = state.get("user_prompt", "")
    file_path = state.get("file_path")

    queue = determine_sub_engines(prompt, file_path)
    engine_models = build_step_models(queue)
    merged_models = {**state.get("step_models", {}), **engine_models}

    print(f"[doc_read] Queued ingestion engines: {queue}")

    return {
        "sub_read_queue": queue,
        "sub_read_type": queue[0] if queue else None,
        "step_models": merged_models,
        "execution_history": [{"node": "doc_read", "status": "queued", "engines": queue}],
    }


def route_doc_read(state: AgentState) -> str:
    queue = state.get("sub_read_queue") or ["ocr"]
    return queue[0]


async def ocr_node(state: AgentState) -> Dict[str, Any]:
    print("--> [OCR] Extracting text from document/scan...")
    await asyncio.sleep(0.05)
    return {
        "extracted_text": (state.get("extracted_text") or "") + "\n[OCR extracted text placeholder]",
        "is_scanned": True,
        "execution_history": [{"node": "ocr", "status": "done"}],
    }


async def rag_node(state: AgentState) -> Dict[str, Any]:
    print("--> [RAG] Retrieving relevant SOP/manual context...")
    await asyncio.sleep(0.05)
    citation = "SOP-1234 sec.3.2"
    return {
        "retrieved_sop_context": "[Relevant SOP excerpt placeholder]",
        "sop_citations": [citation],
        "execution_history": [{"node": "rag", "status": "done", "citation": citation}],
    }


async def vision_node(state: AgentState) -> Dict[str, Any]:
    print("--> [Vision] Interpreting diagram/photo content...")
    await asyncio.sleep(0.05)
    return {
        "extracted_images_summary": "[Vision-model image summary placeholder]",
        "execution_history": [{"node": "vision", "status": "done"}],
    }


async def ingestion_router_node(state: AgentState) -> Dict[str, Any]:
    """Pops the just-finished engine off the queue and decides what's next."""
    queue = list(state.get("sub_read_queue") or [])
    if queue:
        finished = queue.pop(0)
    else:
        finished = None
    print(f"[ingestion_router] finished='{finished}', {len(queue)} engine(s) remaining")
    return {
        "sub_read_queue": queue,
        "sub_read_type": queue[0] if queue else None,
        "execution_history": [{"node": "ingestion_router", "remaining": queue}],
    }


def route_after_ingestion(state: AgentState) -> str:
    queue = state.get("sub_read_queue") or []
    if queue:
        return queue[0]  # more ingestion engines still queued
    plan = state.get("execution_plan") or []
    if "coding_agent" in plan:
        return "coding_agent"
    return "doc_write"


# ---------------------------------------------------------------------------
# Member 5: coding sandbox loop
# ---------------------------------------------------------------------------
async def coding_agent_node(state: AgentState) -> Dict[str, Any]:
    model = state.get("step_models", {}).get("coding_agent", get_model_for_step("coding_agent"))
    print(f"--> [Coding Agent] Generating code with model='{model}'...")
    await asyncio.sleep(0.05)
    return {
        "generated_code": "# placeholder generated code\nresult = 1.24\n",
        "model_name": model,
        "execution_history": [{"node": "coding_agent", "status": "code_generated"}],
    }


async def code_test_agent_node(state: AgentState) -> Dict[str, Any]:
    print("--> [Code Test Agent] Executing in sandbox...")
    await asyncio.sleep(0.05)
    # NOTE: sandbox execution is stubbed here; Member 5 wires in the real
    # runner and should set `passed` from actual execution results.
    passed = True

    update: Dict[str, Any] = {
        "execution_output": "SAFETY_FACTOR:1.24",
        "test_passed": passed,
        "execution_history": [{"node": "code_test_agent", "status": "passed" if passed else "failed"}],
    }
    if not passed:
        update["retry_count"] = state.get("retry_count", 0) + 1
        update["execution_error"] = "Sandbox test failed"
    return update


def route_test_agent(state: AgentState) -> str:
    if state.get("test_passed"):
        return "code_review_agent"
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "coding_agent"
    print("[coding loop] Max retries exceeded at test stage — escalating without a passing test.")
    return "artifact_generator"


async def code_review_agent_node(state: AgentState) -> Dict[str, Any]:
    print("--> [Code Review Agent] Checking correctness/security/quality...")
    await asyncio.sleep(0.05)
    passed = bool(state.get("test_passed"))  # stub: mirrors test result until a real reviewer is wired in

    update: Dict[str, Any] = {
        "review_passed": passed,
        "review_feedback": "Looks good." if passed else "Review failed quality checks.",
        "execution_history": [{"node": "code_review_agent", "status": "passed" if passed else "failed"}],
    }
    if not passed:
        update["retry_count"] = state.get("retry_count", 0) + 1
    return update


def route_review_agent(state: AgentState) -> str:
    if state.get("review_passed"):
        return "doc_write"
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "coding_agent"
    print("[coding loop] Max retries exceeded at review stage — escalating without approval.")
    return "artifact_generator"


# ---------------------------------------------------------------------------
# Member 6: output, validation, guardrails, artifact, human approval
# ---------------------------------------------------------------------------
async def doc_write_node(state: AgentState) -> Dict[str, Any]:
    model = state.get("step_models", {}).get("doc_write", get_model_for_step("doc_write"))
    print(f"--> [Doc Write] Generating report/note with model='{model}'...")
    await asyncio.sleep(0.05)
    return {
        "model_name": model,
        "execution_history": [{"node": "doc_write", "status": "draft_generated"}],
    }


async def output_validator_node(state: AgentState) -> Dict[str, Any]:
    print("--> [Output Validator] Checking required fields/formatting...")
    return {
        "validation_status": "passed",
        "validation_errors": [],
        "execution_history": [{"node": "output_validator", "status": "passed"}],
    }


async def guardrails_check_node(state: AgentState) -> Dict[str, Any]:
    print("--> [Guardrails] Scanning for sensitive info / policy violations...")
    return {
        "guardrails_passed": True,
        "execution_history": [{"node": "guardrails_check", "status": "passed"}],
    }


async def artifact_generator_node(state: AgentState) -> Dict[str, Any]:
    print("--> [Artifact Generator] Producing final deliverable...")
    return {
        "generated_artifact_path": "/mnt/user-data/outputs/final_artifact.docx",
        "execution_history": [{"node": "artifact_generator", "status": "created"}],
    }


async def human_approval_node(state: AgentState) -> Dict[str, Any]:
    artifact = state.get("generated_artifact_path")
    decision = interrupt({"question": "Approve this artifact?", "artifact": artifact})
    approved = bool(decision) if isinstance(decision, bool) else str(decision).lower() in ("yes", "true", "approve")
    print(f"--> [Human Approval] decision={decision} -> approved={approved}")
    return {
        "human_approved": approved,
        "execution_history": [{"node": "human_approval", "approved": approved}],
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("clarification", clarification_node)

    graph.add_node("doc_read", doc_read_node)
    graph.add_node("ocr", ocr_node)
    graph.add_node("rag", rag_node)
    graph.add_node("vision", vision_node)
    graph.add_node("ingestion_router", ingestion_router_node)

    graph.add_node("coding_agent", coding_agent_node)
    graph.add_node("code_test_agent", code_test_agent_node)
    graph.add_node("code_review_agent", code_review_agent_node)

    graph.add_node("doc_write", doc_write_node)
    graph.add_node("output_validator", output_validator_node)
    graph.add_node("guardrails_check", guardrails_check_node)
    graph.add_node("artifact_generator", artifact_generator_node)
    graph.add_node("human_approval", human_approval_node)

    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges("orchestrator", route_orchestrator, {
        "doc_read": "doc_read",
        "coding": "coding_agent",
        "doc_write": "doc_write",
        "clarification": "clarification",
    })
    graph.add_edge("clarification", END)

    graph.add_conditional_edges("doc_read", route_doc_read, {
        "ocr": "ocr", "rag": "rag", "vision": "vision",
    })
    graph.add_edge("ocr", "ingestion_router")
    graph.add_edge("rag", "ingestion_router")
    graph.add_edge("vision", "ingestion_router")
    graph.add_conditional_edges("ingestion_router", route_after_ingestion, {
        "ocr": "ocr", "rag": "rag", "vision": "vision",
        "coding_agent": "coding_agent", "doc_write": "doc_write",
    })

    graph.add_edge("coding_agent", "code_test_agent")
    graph.add_conditional_edges("code_test_agent", route_test_agent, {
        "code_review_agent": "code_review_agent",
        "coding_agent": "coding_agent",
        "artifact_generator": "artifact_generator",
    })
    graph.add_conditional_edges("code_review_agent", route_review_agent, {
        "doc_write": "doc_write",
        "coding_agent": "coding_agent",
        "artifact_generator": "artifact_generator",
    })

    graph.add_edge("doc_write", "output_validator")
    graph.add_edge("output_validator", "guardrails_check")
    graph.add_edge("guardrails_check", "artifact_generator")
    graph.add_edge("artifact_generator", "human_approval")
    graph.add_edge("human_approval", END)

    return graph.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Manual test harness
# ---------------------------------------------------------------------------
async def main():
    app = build_graph()
    config = {"configurable": {"thread_id": "demo-1"}}

    initial_state: AgentState = {
        "user_prompt": "Read this inspection report against SOP-1234 and write an approval note",
        "file_path": "/tmp/inspection.pdf",
        "retry_count": 0,
        "current_step_index": 0,
    }

    result = await app.ainvoke(initial_state, config=config)

    if "__interrupt__" in result:
        print("\n[Paused for human approval] Auto-approving for this demo run...")
        result = await app.ainvoke(Command(resume=True), config=config)

    print("\n--- Final State ---")
    print("route:", result.get("route"))
    print("execution_plan:", result.get("execution_plan"))
    print("step_models:", result.get("step_models"))
    print("error_message:", result.get("error_message"))
    print("generated_artifact_path:", result.get("generated_artifact_path"))
    print("human_approved:", result.get("human_approved"))


if __name__ == "__main__":
    asyncio.run(main())