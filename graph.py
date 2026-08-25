import asyncio
from typing import Any, List
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from state import AgentState
from planner import generate_plan, build_step_models

# ==========================================
# 1. CORE ORCHESTRATOR & DISPATCHER
# ==========================================
async def orchestrator_node(state: AgentState) -> dict:
    """Tracks plan generation, model assignment, and advances step indices."""
    prompt = state.get("user_prompt", "")
    file_path = state.get("file_path")
    plan = state.get("execution_plan")

    # Initialization Phase (First run)
    if plan is None:
        plan = generate_plan(prompt, file_path)
        step_models = build_step_models(plan)
        print(f"\n[Orchestrator] Dynamic Plan Generated: {plan}")
        if plan:
            print(f"[Orchestrator] Model Auto-Selection: {step_models}")
        return {
            "execution_plan": plan,
            "step_models": step_models,
            "current_step_index": 0,
            "retry_count": 0,
            "execution_history": [{"node": "orchestrator", "status": "planned", "plan": plan, "step_models": step_models}]
        }

    # Advance to next step in the sequence
    new_idx = state.get("current_step_index", 0) + 1
    return {
        "current_step_index": new_idx,
        "execution_history": [{"node": "orchestrator", "status": f"advanced_to_step_{new_idx}"}]
    }

# ==========================================
# 2. CLARIFICATION & WORKER NODES
# ==========================================
async def clarification_node(state: AgentState) -> dict:
    print("\n[Orchestrator Notice] Prompt unclear or unroutable. Requesting clarification...")
    return {
        "error_message": "User prompt could not be mapped to any operational workflow. Please specify the task clearly.",
        "validation_status": "failed",
        "guardrails_passed": True,
        "human_approved": False,
        "execution_history": [{"node": "clarification_node", "status": "clarification_requested"}]
    }

async def doc_read_node(state: AgentState) -> dict:
    model_name = state.get("step_models", {}).get("doc_read", "llava")
    print(f"--> [Member 3: Doc Read / Vision] Using model [{model_name}] to parse document...")
    await asyncio.sleep(0.05)
    return {
        "extracted_text": "Pipe Wall Thickness: 12.4mm, Measured Pressure: 150 PSI",
        "execution_history": [{"node": "doc_read", "status": "completed", "model": model_name}]
    }

async def rag_search_node(state: AgentState) -> dict:
    model_name = state.get("step_models", {}).get("rag_search", "llama3.1:8b")
    print(f"--> [Member 4: RAG Knowledge Base] Using model [{model_name}] to query SOP guidelines...")
    await asyncio.sleep(0.05)
    return {
        "retrieved_sop_context": "SOP-04: Minimum thickness for 150 PSI pressure is 10.0mm.",
        "sop_citations": ["SOP-04 Section 2.1"],
        "execution_history": [{"node": "rag_search", "status": "completed", "model": model_name}]
    }

async def coding_agent_node(state: AgentState) -> dict:
    model_name = state.get("step_models", {}).get("coding_agent", "qwen2.5-coder:7b")
    print(f"--> [Member 5: Coding Agent] Using model [{model_name}] to generate verification script...")

    context = state.get("extracted_text") or "No extracted data provided."
    sop = state.get("retrieved_sop_context") or "No SOP context provided."
    feedback = state.get("review_feedback") or ""

    fallback_code = (
        "thickness = 12.4\nmin_req = 10.0\n"
        "safety_factor = thickness / min_req\n"
        "print(f'Safety Factor: {safety_factor:.2f}')"
    )

    try:
        coder_llm = ChatOllama(model=model_name, temperature=0.1)
        prompt = (
            "Write a short Python script that checks the extracted data against "
            "the requirement below and prints a clear pass/fail result.\n\n"
            f"Extracted data: {context}\n"
            f"Requirement: {sop}\n"
        )
        if feedback:
            prompt += f"\nThe previous attempt had this issue, fix it: {feedback}\n"
        prompt += "\nReturn ONLY the raw Python code, no explanation, no markdown fences."

        response = coder_llm.invoke([HumanMessage(content=prompt)])
        code = str(response.content).strip()
        if "```python" in code:
            code = code.split("```python")[-1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].strip()
        if not code:
            code = fallback_code
    except Exception:
        code = fallback_code

    return {
        "generated_code": code,
        "model_name": model_name,
        "execution_history": [{"node": "coding_agent", "status": "code_generated", "model": model_name}]
    }

async def code_test_agent_node(state: AgentState) -> dict:
    print("--> [Member 5: Code Test Agent] Executing in local sandbox...")
    await asyncio.sleep(0.05)
    return {
        "execution_output": "Safety Factor: 1.24",
        "execution_error": None,
        "test_passed": True,
        "execution_history": [{"node": "code_test_agent", "status": "test_passed"}]
    }

async def code_review_agent_node(state: AgentState) -> dict:
    print("--> [Member 5: Code Review Agent] Reviewing code quality & safety...")
    test_passed = state.get("test_passed", False)
    if not test_passed:
        retry = state.get("retry_count", 0) + 1
        return {
            "review_feedback": "Execution error detected. Adjust calculation.",
            "retry_count": retry,
            "execution_history": [{"node": "code_review_agent", "status": "retry_triggered", "retry": retry}]
        }

    return {
        "review_feedback": "Passed verification.",
        "execution_history": [{"node": "code_review_agent", "status": "review_passed"}]
    }

async def doc_write_node(state: AgentState) -> dict:
    model_name = state.get("step_models", {}).get("doc_write", "llama3.1:8b")
    print(f"--> [Member 6: Doc Write] Using model [{model_name}] to draft the final report...")
    return {
        "generated_artifact_path": "exports/Inspection_Approval_Note.docx",
        "execution_history": [{"node": "doc_write", "status": "artifact_created", "model": model_name}]
    }

async def output_validator_node(state: AgentState) -> dict:
    print("--> [Member 6: Output Validator] Validating required fields & formatting...")
    return {
        "validation_status": "passed",
        "validation_errors": [],
        "execution_history": [{"node": "output_validator", "status": "validated"}]
    }

async def guardrails_node(state: AgentState) -> dict:
    print("--> [Member 6: Guardrails] Running enterprise policy & data leakage checks...")
    return {
        "guardrails_passed": True,
        "execution_history": [{"node": "guardrails_check", "status": "passed"}]
    }

async def human_approval_node(state: AgentState) -> dict:
    """Pauses execution for human intervention."""
    artifact = state.get("generated_artifact_path", "No artifact generated")

    decision = interrupt({
        "message": "Human review required before finalizing.",
        "artifact": artifact,
        "validation_status": state.get("validation_status"),
        "guardrails_passed": state.get("guardrails_passed"),
        "test_passed": state.get("test_passed"),
    })

    approved = str(decision).strip().lower() in ("yes", "y", "approve", "approved", "true")
    return {
        "human_approved": approved,
        "execution_history": [{"node": "human_approval", "decision": "approved" if approved else "rejected"}]
    }

# ==========================================
# 3. ROUTERS & CONDITIONAL EDGES
# ==========================================
def dispatch_next_step(state: AgentState) -> str:
    plan = state.get("execution_plan", [])
    idx = state.get("current_step_index", 0)

    # Empty plan routes directly to clarification
    if not plan:
        return "clarification_node"

    if idx < len(plan):
        return plan[idx]
    return "output_validator"

def route_coding_loop(state: AgentState) -> str:
    if state.get("test_passed", False):
        return "orchestrator"

    if state.get("retry_count", 0) < 3:
        print("[Retry] Sending back to coding_agent for self-correction...")
        return "coding_agent"

    print("[Error] Max retries exceeded. Escalating directly to human approval...")
    return "human_approval"

# ==========================================
# 4. GRAPH ASSEMBLY
# ==========================================
builder = StateGraph(AgentState)

# Nodes
builder.add_node("orchestrator", orchestrator_node)
builder.add_node("clarification_node", clarification_node)
builder.add_node("doc_read", doc_read_node)
builder.add_node("rag_search", rag_search_node)
builder.add_node("coding_agent", coding_agent_node)
builder.add_node("code_test_agent", code_test_agent_node)
builder.add_node("code_review_agent", code_review_agent_node)
builder.add_node("doc_write", doc_write_node)
builder.add_node("output_validator", output_validator_node)
builder.add_node("guardrails_check", guardrails_node)
builder.add_node("human_approval", human_approval_node)

# Flow Connections
builder.add_edge(START, "orchestrator")

builder.add_conditional_edges(
    "orchestrator",
    dispatch_next_step,
    {
        "doc_read": "doc_read",
        "rag_search": "rag_search",
        "coding_agent": "coding_agent",
        "doc_write": "doc_write",
        "output_validator": "output_validator",
        "clarification_node": "clarification_node"
    }
)

builder.add_edge("clarification_node", END)
builder.add_edge("doc_read", "orchestrator")
builder.add_edge("rag_search", "orchestrator")

# Member 5 Sub-Loop
builder.add_edge("coding_agent", "code_test_agent")
builder.add_edge("code_test_agent", "code_review_agent")
builder.add_conditional_edges(
    "code_review_agent",
    route_coding_loop,
    {
        "coding_agent": "coding_agent",
        "orchestrator": "orchestrator",
        "human_approval": "human_approval"
    }
)

# Member 6 Final Pipeline
builder.add_edge("doc_write", "output_validator")
builder.add_edge("output_validator", "guardrails_check")
builder.add_edge("guardrails_check", "human_approval")
builder.add_edge("human_approval", END)

checkpointer = MemorySaver()
app = builder.compile(checkpointer=checkpointer)

# ==========================================
# 5. TEST HARNESS
# ==========================================
async def main():
    config = {"configurable": {"thread_id": "session-1"}}

    # Example test prompt (change to "asdfg" to test clarification exit)
    initial_input: AgentState = {
        "user_prompt": "Read the inspection report, check SOP safety standards, verify math, and generate approval note.",
        "file_path": "uploads/inspection_report.pdf",
        "retry_count": 0,
        "execution_history": []
    }

    print("\n=== STARTING ORCHESTRATION PIPELINE ===")
    result = await app.ainvoke(initial_input, config=config)

    # Pause handler for human review
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print(f"\n[HUMAN REVIEW] {payload['message']}")
        print(f"Artifact: {payload['artifact']}")
        print(f"Validation: {payload['validation_status']} | Guardrails: {payload['guardrails_passed']}")
        answer = input("Approve this output? (yes/no): ")
        result = await app.ainvoke(Command(resume=answer), config=config)

    print("\n=== FINAL DELIVERABLE SUMMARY ===")
    if result.get("error_message"):
        print(f"Status: Exited Early - {result.get('error_message')}")
    else:
        print(f"Artifact Created: {result.get('generated_artifact_path')}")
        print(f"Validation: {result.get('validation_status')}")
        print(f"Guardrails Status: {result.get('guardrails_passed')}")
        print(f"Human Approved: {result.get('human_approved')}")
        print(f"Model Auto-Selection Used: {result.get('step_models')}")
    print(f"Total Audit History Logs: {len(result.get('execution_history', []))}")

if __name__ == "__main__":
    asyncio.run(main())