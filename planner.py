import os
import re
import json
from typing import Optional, List, Dict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

ROUTER_MODEL = os.getenv("ROUTER_MODEL", "llama3.1:8b")
router_llm = ChatOllama(model=ROUTER_MODEL, temperature=0)

STEP_MODEL_MAP: Dict[str, str] = {
    "doc_read": os.getenv("VISION_MODEL", "llava"),
    "rag_search": os.getenv("RAG_MODEL", "llama3.1:8b"),
    "coding_agent": os.getenv("CODING_MODEL", "qwen2.5-coder:7b"),
    "doc_write": os.getenv("DOC_MODEL", "llama3.1:8b"),
}

def get_model_for_step(step_name: str) -> str:
    return STEP_MODEL_MAP.get(step_name, ROUTER_MODEL)

def build_step_models(plan: List[str]) -> Dict[str, str]:
    return {step: get_model_for_step(step) for step in plan}

# --- LEVEL 1: REGEX DETERMINISTIC PARSER (<1ms) ---
def level_1_regex_planner(prompt: str, has_file: bool) -> Optional[List[str]]:
    text = (prompt or "").strip().lower()

    # Reject obvious gibberish or empty prompts
    if not text or len(text) < 3 or re.fullmatch(r"[^a-zA-Z0-9\s]+", text):
        return []
    if re.fullmatch(r"(.)\1{3,}", text) or text in ["asdf", "asdfg", "asdfgh", "qwerty", "zxcv"]:
        return []

    plan: List[str] = []
    
    # 1. Check document ingestion requirement
    if has_file or re.search(r"\b(read|parse|extract|scan|image|pdf|drawing|blueprint|ocr|vision)\b", text):
        plan.append("doc_read")

    # 2. Check enterprise RAG lookup
    if re.search(r"\b(sop|standard|guideline|manual|policy|compliance|clause|rag)\b", text):
        plan.append("rag_search")

    # 3. Check math / coding verification
    if re.search(r"\b(calculate|verify|math|safety factor|formula|compute|code|script)\b", text):
        plan.append("coding_agent")

    # 4. Check report/writing requirement
    if re.search(r"\b(report|write|summary|document|note|dossier|draft|approval|export|essay)\b", text):
        plan.append("doc_write")

    # If Regex identified multiple structured steps or explicit doc_write
    if len(plan) >= 2 or (len(plan) == 1 and plan[0] == "doc_write"):
        print("--> [Planner Tier 1] Plan resolved via Regex pattern matching.")
        return plan

    return None

# --- LEVEL 2: LLM SEMANTIC STEP DECOMPOSER ---
PLANNER_SYSTEM_PROMPT = """
You are the central Orchestrator Brain for a Sovereign AI system.
Break down the user task into an ordered JSON list of execution steps from these worker nodes:
- "doc_read": Document text, image, or blueprint parsing.
- "rag_search": Query company SOPs, manuals, or engineering standards.
- "coding_agent": Run calculations, write/execute Python code in a sandbox.
- "doc_write": Generate final Word, Excel, or approval notes.

Rules:
1. If the prompt is unclear or nonsensical, return {"plan": []}.
2. Return ONLY valid JSON: {"plan": ["step1", "step2"]}
"""

def generate_execution_plan(user_prompt: str, file_path: Optional[str] = None) -> List[str]:
    has_file = bool(file_path and file_path.strip())

    # Level 1 Check
    quick_plan = level_1_regex_planner(user_prompt, has_file)
    if quick_plan is not None:
        return quick_plan

    # Level 2 Fallback
    print("--> [Planner Tier 2] Conversational prompt. Delegating to LLM semantic planner...")
    try:
        response = router_llm.invoke([
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"User Task: {user_prompt}\nFile Attached: {has_file}")
        ])
        content = str(response.content).strip()
        if "```json" in content:
            content = content.split("```json")[-1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()

        data = json.loads(content)
        plan = data.get("plan", [])
        return plan if isinstance(plan, list) else []
    except Exception:
        # Default to clarification if unrecognized to prevent wrong outputs
        return []