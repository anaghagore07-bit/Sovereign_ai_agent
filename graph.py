import asyncio
import re
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from planner import generate_execution_plan, build_step_models

# ==========================================
# 1. WATCHMAN & ORCHESTRATOR
# ==========================================
async def watchman_node(state: AgentState) -> dict:
    file_path = state.get("file_path")
    status = f"Ingested file: {file_path}" if file_path else "Direct prompt intake"
    print(f"\n[Watchman] Intake: {status}")
    return {"execution_history": [{"node": "watchman", "status": status}]}

async def orchestrator_node(state: AgentState) -> dict:
    plan = state.get("execution_plan")

    # Initial Planning Phase
    if plan is None:
        prompt = state.get("user_prompt", "")
        file_path = state.get("file_path")
        plan = generate_execution_plan(prompt, file_path)
        step_models = build_step_models(plan)

        print(f"[Orchestrator] Dynamic Plan Generated: {plan}")
        if plan:
            print(f"[Orchestrator] Model Auto-Selection: {step_models}")

        return {
            "execution_plan": plan,
            "step_models": step_models,
            "current_step_index": 0,
            "retry_count": 0,
            "execution_history": [{"node": "orchestrator", "status": "plan_initialized", "plan": str(plan)}]
        }

    # Step Advancement Phase
    next_idx = state.get("current_step_index", 0) + 1
    return {
        "current_step_index": next_idx,
        "execution_history": [{"node": "orchestrator", "status": f"advanced_to_step_{next_idx}"}]
    }

async def clarification_node(state: AgentState) -> dict:
    print("[Orchestrator Notice] Request unclear or unroutable. Clarification requested.")
    return {
        "error_message": "Prompt could not be mapped to any operational workflow. Please specify the task clearly.",
        "validation_status": "failed",
        "guardrails_passed": True,
        "human_approved": False,
        "execution_history": [{"node": "clarification_node", "status": "clarification_requested"}]
    }

# ==========================================
# 2. WORKER NODES (CONSUMING AUTO-SELECTED MODELS)
# ==========================================
async def doc_read_node(state: AgentState) -> dict:
    model = state.get("step_models", {}).get("doc_read", "llava")
    print(f"--> [Member 3: Doc Read / Vision] Using assigned model [{model}] to parse document...")
    await asyncio.sleep(0.05)
    return {
        "extracted_text": "Parsed Spec: Wall Thickness = 12.4mm, Measured Pressure = 150 PSI",
        "execution_history": [{"node": "doc_read", "status": "completed", "model": model}]
    }

async def rag_search_node(state: AgentState) -> dict:
    model = state.get("step_models", {}).get("rag_search", "llama3.1:8b")
    print(f"--> [Member 4: RAG Engine] Using assigned model [{model}] to query SOP knowledge base...")
    await asyncio.sleep(0.05)
    return {
        "retrieved_context": "SOP Standard Clause 4.2: Required thickness threshold for 150 PSI is 10.0mm.",
        "sop_citations": ["SOP Clause 4.2"],
        "execution_history": [{"node": "rag_search", "status": "completed", "model": model}]
    }

async def coding_agent_node(state: AgentState) -> dict:
    model = state.get("step_models", {}).get("coding_agent", "qwen2.5-coder:7b")
    print(f"--> [Member 5: Coding Agent] Using assigned model [{model}] to generate calculation script...")
    
    context = state.get("extracted_text") or ""
    val_match = re.findall(r"(\d+(?:\.\d+)?)", context)
    thickness = float(val_match[0]) if val_match else 12.4
    
    code = (
        f"thickness = {thickness}\n"
        "min_req = 10.0\n"
        "safety_factor = thickness / min_req\n"
        "print(f'SAFETY_FACTOR:{safety_factor:.2f}')"
    )
    return {
        "generated_code": code,
        "execution_history": [{"node": "coding_agent", "status": "code_generated", "model": model}]
    }

async def code_test_agent_node(state: AgentState) -> dict:
    print("--> [Member 5: Code Test Agent] Executing in local sandbox...")
    await asyncio.sleep(0.05)
    # Simulated execution output
    test_ok = True  
    retries = state.get("retry_count", 0) if test_ok else state.get("retry_count", 0) + 1
    
    return {
        "execution_output": "SAFETY_FACTOR:1.24",
        "test_passed": test_ok,
        "retry_count": retries,
        "execution_history": [{"node": "code_test_agent", "status": "passed" if test_ok else "failed"}]
    }

async def code_review_agent_node(state: AgentState) -> dict:
    print("--> [Member 5: Code Review Agent] Reviewing code quality & safety...")
    test_passed = state.get("test_passed", False)
    retries = state.get("retry_count", 0) if test_passed else state.get("retry_count", 0) + 1
    
    return {
        "review_passed": test_passed,
        "retry_count": retries,
        "execution_history": [{"node": "code_review_agent", "status": "approved" if test_passed else "rejected"}]
    }

async def doc_write_node(state: AgentState) -> dict:
    model = state.get("step_models", {}).get("doc_write", "llama3.1:8b")
    print(f"--> [Member 6: Doc Write] Using assigned model [{model}] to draft deliverable...")
    return {
        "artifact_path": "exports/Engineering_Compliance_Report.docx",
        "execution_history": [{"node": "doc_write", "status": "report_drafted", "model": model}]
    }

async def output_validator_node(state: AgentState) -> dict:
    print("--> [Member 6: Output Validator] Verifying schemas and field completeness...")
    return {
        "validation_status": "passed",
        "execution_history": [{"node": "output_validator", "status": "validated"}]
    }

async def guardrails_check_node(state: AgentState) -> dict:
    print("--> [Member 6: Guardrails Check] Scanning for policy violations & data leakage...")
    return {
        "guardrails_passed": True,
        "execution_history": [{"node": "guardrails_check", "status": "passed"}]
    }

async def human_approval_node(state: AgentState) -> dict:
    artifact = state.get("artifact_path", "No artifact")
    decision = interrupt({
        "message": "Human approval required for final release.",
        "artifact": artifact,
        "validation": state.get("validation_status"),
        "guardrails": state.get("guardrails_passed")
    })
    approved = str(decision).strip().lower() in ("yes", "y", "approve", "approved")
    return {
        "human_approved": approved,
        "execution_history": [{"node": "human_approval", "status": "approved" if approved else "rejected"}]
    }

# ==========================================
# 3. SEQUENTIAL DISPATCHER & CONDITIONAL EDGES
# ==========================================
def dispatch_next_step(state: AgentState) -> str:
    plan = state.get("execution_plan", [])
    idx = state.get("current_step_index", 0)

    if not plan:
        return "clarification_node"
    if idx < len(plan):
        return plan[idx]
    
    # All plan steps complete -> enter delivery/validation gate
    return "output_validator"

def route_coding_loop(state: AgentState) -> str:
    if state.get("review_passed", False):
        return "orchestrator"
    
    if state.get("retry_count", 0) < 3:
        print(f"[Retry Triggered] Attempt {state.get('retry_count')}/3. Correcting code...")
        return "coding_agent"

    print("[Error] Max retries reached. Escalating to output validator...")
    return "output_validator"

# ==========================================
# 4. GRAPH ASSEMBLY
# ==========================================
builder = StateGraph(AgentState)

# Nodes
builder.add_node("watchman", watchman_node)
builder.add_node("orchestrator", orchestrator_node)
builder.add_node("clarification_node", clarification_node)
builder.add_node("doc_read", doc_read_node)
builder.add_node("rag_search", rag_search_node)
builder.add_node("coding_agent", coding_agent_node)
builder.add_node("code_test_agent", code_test_agent_node)
builder.add_node("code_review_agent", code_review_agent_node)
builder.add_node("doc_write", doc_write_node)
builder.add_node("output_validator", output_validator_node)
builder.add_node("guardrails_check", guardrails_check_node)
builder.add_node("human_approval", human_approval_node)

# Flow Connections
builder.add_edge(START, "watchman")
builder.add_edge("watchman", "orchestrator")

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

# Coding Subgraph
builder.add_edge("coding_agent", "code_test_agent")
builder.add_edge("code_test_agent", "code_review_agent")
builder.add_conditional_edges(
    "code_review_agent",
    route_coding_loop,
    {
        "coding_agent": "coding_agent",
        "orchestrator": "orchestrator",
        "output_validator": "output_validator"
    }
)

# Document Validation, Guardrails & Human Gate
builder.add_edge("doc_write", "orchestrator")
builder.add_edge("output_validator", "guardrails_check")
builder.add_edge("guardrails_check", "human_approval")
builder.add_edge("human_approval", END)

checkpointer = MemorySaver()
app = builder.compile(checkpointer=checkpointer)

# ==========================================
# 5. TEST HARNESS
# ==========================================
async def main():
    config = {"configurable": {"thread_id": "final_orchestrator_session"}}

    initial_input: AgentState = {
        "user_prompt": "Read the inspection report, check SOP safety standards, verify math, and generate approval note.",
        "file_path": "uploads/inspection_report.pdf",
        "retry_count": 0,
        "execution_history": []
    }

    print("\n=== STARTING ORCHESTRATION PIPELINE ===")
    result = await app.ainvoke(initial_input, config=config)

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print(f"\n[HUMAN REVIEW REQUIRED] {payload['message']}")
        print(f"Artifact: {payload['artifact']}")
        print(f"Validation: {payload['validation']} | Guardrails: {payload['guardrails']}")
        
        while True:
            choice = input("Approve this deliverable? (yes/no): ").strip().lower()
            if choice in ("yes", "y", "no", "n"):
                break
            print("Please type 'yes' or 'no'.")

        result = await app.ainvoke(Command(resume=choice), config=config)

    print("\n=== DELIVERABLE SUMMARY ===")
    if result.get("error_message"):
        print(f"Status: Exited Early - {result.get('error_message')}")
    else:
        print(f"Artifact Created: {result.get('artifact_path')}")
        print(f"Validation: {result.get('validation_status')}")
        print(f"Guardrails: {result.get('guardrails_passed')}")
        print(f"Human Approved: {result.get('human_approved')}")
        print(f"Model Auto-Selection: {result.get('step_models')}")
    print(f"Total Audit History Logs: {len(result.get('execution_history', []))}")

if __name__ == "__main__":
    asyncio.run(main())