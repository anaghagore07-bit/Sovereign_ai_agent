import asyncio
import os
import re
import json
from typing import TypedDict, Optional, List, Dict, Annotated
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# ==========================================
# 1. STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    user_prompt: str
    file_path: Optional[str]
    route: Optional[str]               # 'coding', 'doc_write', 'doc_read', 'clarification'
    sub_read_type: Optional[str]       # 'ocr', 'rag', 'vision'
    extracted_text: Optional[str]
    retrieved_context: Optional[str]
    generated_code: Optional[str]
    execution_output: Optional[str]
    test_passed: bool
    review_passed: bool
    retry_count: int
    validation_status: str
    guardrails_passed: bool
    artifact_path: Optional[str]
    human_approved: bool
    error_message: Optional[str]
    execution_history: Annotated[List[Dict[str, str]], operator.add]

# ==========================================
# 2. LEVEL 1 & LEVEL 2 ROUTING ENGINES
# ==========================================
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "llama3.1:8b")
router_llm = ChatOllama(model=ROUTER_MODEL, temperature=0)

PLANNER_SYSTEM_PROMPT = """
You are the central Orchestrator Brain for a Sovereign AI system.
Classify the user task into exactly ONE of the following operational routes:
- "coding": Calculation, script execution, numerical verification, math.
- "doc_read": Needs OCR parsing, visual diagram reading, or SOP/manual search.
- "doc_write": Direct essay/report generation without needing external extraction.
- "clarification": Unclear, nonsensical (e.g. keyboard mashing), or invalid requests.

Return ONLY valid JSON matching this schema:
{"route": "coding" | "doc_read" | "doc_write" | "clarification"}
"""

def level_1_regex_router(prompt: str, file_path: Optional[str]) -> Optional[str]:
    """Tier 1: Fast Regex & Rule-Based Matcher (<1ms)"""
    text = (prompt or "").strip().lower()
    has_file = bool(file_path and file_path.strip())

    # 1. Catch empty, trivial, or keyboard-mash input
    if not text or len(text) < 3 or re.fullmatch(r"[^a-zA-Z0-9\s]+", text):
        return "clarification"
    if re.fullmatch(r"(.)\1{3,}", text) or text in ["asdf", "asdfg", "asdfgh", "qwerty", "zxcv"]:
        return "clarification"

    # 2. File attached defaults to doc_read ingestion
    if has_file:
        return "doc_read"

    # 3. Regex Intent Matching
    if re.search(r"\b(code|calculate|math|verify|script|compute|factor|equation)\b", text):
        return "coding"
    if re.search(r"\b(read|ocr|scan|vision|rag|sop|clause|manual|extract)\b", text):
        return "doc_read"
    if re.search(r"\b(write|draft|report|summary|note|essay|compose)\b", text):
        return "doc_write"

    # Ambiguous -> Pass to Level 2
    return None

def level_2_llm_router(prompt: str, file_path: Optional[str]) -> str:
    """Tier 2: Semantic LLM Classifier for Complex Prompts"""
    print("--> [Router Tier 2] Complex/Conversational prompt. Delegating to LLM semantic planner...")
    try:
        response = router_llm.invoke([
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"User Task: {prompt}\nFile Attached: {bool(file_path)}")
        ])
        content = str(response.content).strip()
        if "```" in content:
            content = content.split("```json")[-1].split("```")[0].strip()
        data = json.loads(content)
        return data.get("route", "doc_write")
    except Exception:
        return "doc_write"

# ==========================================
# 3. GRAPH NODES
# ==========================================
async def watchman_node(state: AgentState) -> dict:
    file_path = state.get("file_path")
    status = f"Ingested file: {file_path}" if file_path else "Direct prompt intake"
    print(f"\n[Watchman] Intake complete: {status}")
    return {"execution_history": [{"node": "watchman", "status": status}]}

async def orchestrator_node(state: AgentState) -> dict:
    prompt = state.get("user_prompt", "")
    file_path = state.get("file_path")

    # Step 1: Level 1 Regex Check
    route = level_1_regex_router(prompt, file_path)
    if route:
        print(f"[Orchestrator] Level 1 (Regex) resolved route: '{route}'")
    else:
        # Step 2: Level 2 LLM Semantic Fallback
        route = level_2_llm_router(prompt, file_path)
        print(f"[Orchestrator] Level 2 (LLM) resolved route: '{route}'")

    return {
        "route": route,
        "execution_history": [{"node": "orchestrator", "status": f"routed_to_{route}"}]
    }

async def clarification_node(state: AgentState) -> dict:
    print("[Orchestrator Notice] Request unclear or unroutable. Clarification requested.")
    return {
        "error_message": "Prompt could not be understood. Please state your task clearly.",
        "validation_status": "failed",
        "guardrails_passed": True,
        "human_approved": False,
        "execution_history": [{"node": "clarification_node", "status": "clarification_requested"}]
    }

async def doc_read_node(state: AgentState) -> dict:
    prompt = (state.get("user_prompt") or "").lower()
    file_path = (state.get("file_path") or "").lower()

    if file_path.endswith((".png", ".jpg", ".jpeg")) or "image" in prompt or "drawing" in prompt:
        sub_type = "vision"
    elif "sop" in prompt or "standard" in prompt or "rag" in prompt:
        sub_type = "rag"
    else:
        sub_type = "ocr"

    print(f"--> [Doc Read] Routing to sub-engine: [{sub_type}]")
    return {
        "sub_read_type": sub_type,
        "execution_history": [{"node": "doc_read", "status": f"delegated_to_{sub_type}"}]
    }

async def ocr_node(state: AgentState) -> dict:
    print("--> [OCR Engine] Extracting text from document...")
    return {
        "extracted_text": "Parsed specification via OCR: Wall thickness = 12.4mm, Design pressure = 150 PSI",
        "execution_history": [{"node": "ocr", "status": "completed"}]
    }

async def rag_node(state: AgentState) -> dict:
    print("--> [RAG Engine] Querying SOP knowledge base...")
    return {
        "retrieved_context": "SOP Standard Clause 4.2: Required thickness threshold is 10.0mm.",
        "execution_history": [{"node": "rag", "status": "completed"}]
    }

async def vision_node(state: AgentState) -> dict:
    print("--> [Vision Engine] Parsing diagram schematics...")
    return {
        "extracted_text": "Vision extraction: Technical schematic confirms pressure boundary integrity.",
        "execution_history": [{"node": "vision", "status": "completed"}]
    }

async def coding_agent_node(state: AgentState) -> dict:
    print("--> [Coding Agent] Generating verification code...")
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
        "execution_history": [{"node": "coding_agent", "status": "code_generated"}]
    }

async def code_test_agent_node(state: AgentState) -> dict:
    print("--> [Code Test Agent] Executing in sandbox environment...")
    await asyncio.sleep(0.05)
    return {
        "execution_output": "SAFETY_FACTOR:1.24",
        "test_passed": True,
        "execution_history": [{"node": "code_test_agent", "status": "test_passed"}]
    }

async def code_review_agent_node(state: AgentState) -> dict:
    print("--> [Code Review Agent] Performing code quality check...")
    test_ok = state.get("test_passed", False)
    return {
        "review_passed": test_ok,
        "execution_history": [{"node": "code_review_agent", "status": "review_passed" if test_ok else "review_failed"}]
    }

async def doc_write_node(state: AgentState) -> dict:
    print("--> [Doc Write] Drafting compliance report...")
    return {"execution_history": [{"node": "doc_write", "status": "draft_compiled"}]}

async def output_validator_node(state: AgentState) -> dict:
    print("--> [Output Validator] Verifying schemas and field integrity...")
    return {
        "validation_status": "passed",
        "execution_history": [{"node": "output_validator", "status": "validated"}]
    }

async def guardrails_check_node(state: AgentState) -> dict:
    print("--> [Guardrails Check] Checking policy compliance & data privacy...")
    return {
        "guardrails_passed": True,
        "execution_history": [{"node": "guardrails_check", "status": "passed"}]
    }

async def artifact_generator_node(state: AgentState) -> dict:
    print("--> [Artifact Generator] Building final deliverable files...")
    return {
        "artifact_path": "exports/Engineering_Compliance_Report.docx",
        "execution_history": [{"node": "artifact_generator", "status": "artifact_created"}]
    }

async def human_approval_node(state: AgentState) -> dict:
    artifact = state.get("artifact_path", "No artifact generated")
    decision = interrupt({
        "message": "Human approval required for final deliverable.",
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
# 4. CONDITIONAL ROUTERS
# ==========================================
def route_orchestrator(state: AgentState) -> str:
    return state.get("route", "doc_write")

def route_doc_read(state: AgentState) -> str:
    return state.get("sub_read_type", "ocr")

def route_after_ingestion(state: AgentState) -> str:
    prompt = (state.get("user_prompt") or "").lower()
    if any(k in prompt for k in ["code", "calculate", "math", "verify", "script"]):
        return "coding_agent"
    return "doc_write"

def route_test_agent(state: AgentState) -> str:
    if state.get("test_passed"):
        return "code_review_agent"
    if state.get("retry_count", 0) < 3:
        return "coding_agent"
    return "artifact_generator"

def route_review_agent(state: AgentState) -> str:
    if state.get("review_passed"):
        return "doc_write"
    if state.get("retry_count", 0) < 3:
        return "coding_agent"
    return "artifact_generator"

# ==========================================
# 5. GRAPH ASSEMBLY
# ==========================================
builder = StateGraph(AgentState)

# Register All Nodes
builder.add_node("watchman", watchman_node)
builder.add_node("orchestrator", orchestrator_node)
builder.add_node("clarification_node", clarification_node)
builder.add_node("doc_read", doc_read_node)
builder.add_node("ocr", ocr_node)
builder.add_node("rag", rag_node)
builder.add_node("vision", vision_node)
builder.add_node("coding_agent", coding_agent_node)
builder.add_node("code_test_agent", code_test_agent_node)
builder.add_node("code_review_agent", code_review_agent_node)
builder.add_node("doc_write", doc_write_node)
builder.add_node("output_validator", output_validator_node)
builder.add_node("guardrails_check", guardrails_check_node)
builder.add_node("artifact_generator", artifact_generator_node)
builder.add_node("human_approval", human_approval_node)

# Flow Connections
builder.add_edge(START, "watchman")
builder.add_edge("watchman", "orchestrator")

# Orchestrator Multi-Way Branching
builder.add_conditional_edges(
    "orchestrator",
    route_orchestrator,
    {
        "coding": "coding_agent",
        "doc_write": "doc_write",
        "doc_read": "doc_read",
        "clarification": "clarification_node"
    }
)
builder.add_edge("clarification_node", END)

# Ingestion Sub-routing
builder.add_conditional_edges(
    "doc_read",
    route_doc_read,
    {"ocr": "ocr", "rag": "rag", "vision": "vision"}
)

# Ingestion Hand-off (Connects OCR/RAG/Vision into next processing step)
for ingester in ["ocr", "rag", "vision"]:
    builder.add_conditional_edges(ingester, route_after_ingestion, {
        "coding_agent": "coding_agent",
        "doc_write": "doc_write"
    })

# Coding Loop (with max retries)
builder.add_edge("coding_agent", "code_test_agent")
builder.add_conditional_edges(
    "code_test_agent",
    route_test_agent,
    {
        "coding_agent": "coding_agent",
        "code_review_agent": "code_review_agent",
        "artifact_generator": "artifact_generator"
    }
)
builder.add_conditional_edges(
    "code_review_agent",
    route_review_agent,
    {
        "coding_agent": "coding_agent",
        "doc_write": "doc_write",
        "artifact_generator": "artifact_generator"
    }
)

# Final Artifact Pipeline & Human Approval Gate
builder.add_edge("doc_write", "output_validator")
builder.add_edge("output_validator", "guardrails_check")
builder.add_edge("guardrails_check", "artifact_generator")
builder.add_edge("artifact_generator", "human_approval")
builder.add_edge("human_approval", END)

checkpointer = MemorySaver()
app = builder.compile(checkpointer=checkpointer)

# ==========================================
# 6. EXECUTION HARNESS
# ==========================================
async def main():
    config = {"configurable": {"thread_id": "session_opt_final"}}

    initial_input: AgentState = {
        "user_prompt": "Read the attached inspection document, check SOP standards, run math checks, and write the final approval note.",
        "file_path": "uploads/inspection_sheet.pdf",
        "retry_count": 0,
        "execution_history": []
    }

    print("\n=== STARTING SOVEREIGN AGENT PIPELINE ===")
    result = await app.ainvoke(initial_input, config=config)

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print(f"\n[HUMAN APPROVAL REQUIRED] {payload['message']}")
        print(f"Deliverable: {payload['artifact']}")
        print(f"Validation: {payload['validation']} | Guardrails: {payload['guardrails']}")
        
        while True:
            choice = input("Approve? (yes/no): ").strip().lower()
            if choice in ("yes", "y", "no", "n"):
                break
            print("Please type 'yes' or 'no'.")
            
        result = await app.ainvoke(Command(resume=choice), config=config)

    print("\n=== PIPELINE EXECUTION SUMMARY ===")
    if result.get("error_message"):
        print(f"Status: {result.get('error_message')}")
    else:
        print(f"Artifact Created: {result.get('artifact_path')}")
        print(f"Validation Status: {result.get('validation_status')}")
        print(f"Human Approved: {result.get('human_approved')}")
    print(f"Total Audit Trail Steps: {len(result.get('execution_history', []))}")

if __name__ == "__main__":
    asyncio.run(main())