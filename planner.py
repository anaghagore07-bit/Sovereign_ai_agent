import os
import json
from typing import Optional, List, Dict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

MODEL_NAME = os.getenv("ROUTER_MODEL", "llama3.1:8b")
llm = ChatOllama(model=MODEL_NAME, temperature=0)

STEP_MODEL_MAP: Dict[str, str] = {
    "doc_read": os.getenv("VISION_MODEL", "llava"),
    "rag_search": os.getenv("ROUTER_MODEL", "llama3.1:8b"),
    "coding_agent": os.getenv("CODING_MODEL", "qwen2.5-coder:7b"),
    "doc_write": os.getenv("ROUTER_MODEL", "llama3.1:8b"),
}

def get_model_for_step(step_name: str) -> str:
    return STEP_MODEL_MAP.get(step_name, MODEL_NAME)

def build_step_models(plan: List[str]) -> Dict[str, str]:
    return {step: get_model_for_step(step) for step in plan}

PLANNER_SYSTEM_PROMPT = """
You are the central Orchestrator Brain for a Sovereign AI system.
Analyze the user prompt and generate an ordered JSON execution plan using available worker nodes:
- "doc_read": Parse PDFs, scanned images, diagrams.
- "rag_search": Query company SOPs, manuals, or engineering standards.
- "coding_agent": Run calculations, write/execute Python code in a sandbox.
- "doc_write": Generate final Word, Excel, or approval reports.

Rules:
1. If the user prompt is completely unclear, nonsensical, or cannot be mapped to any task, return {"plan": []}.
2. Otherwise, return ONLY valid JSON matching this schema:
{"plan": ["node_1", "node_2"]}
"""

def generate_plan(user_prompt: str, file_path: Optional[str] = None) -> List[str]:
    # Quick sanity check for empty or trivial inputs
    if not user_prompt or len(user_prompt.strip()) < 4:
        return []

    try:
        response = llm.invoke([
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"User Task: {user_prompt}\nFile Attached: {file_path is not None}")
        ])
        content = str(response.content).strip()
        if "```" in content:
            content = content.split("```json")[-1].split("```")[0].strip()

        data = json.loads(content)
        plan = data.get("plan", [])
        return plan if isinstance(plan, list) else []
    except Exception:
        # If model is offline but input has reasonable length, provide a standard fallback
        return ["doc_read", "rag_search", "coding_agent", "doc_write"] if file_path else ["rag_search", "coding_agent", "doc_write"]