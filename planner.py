import os
import json
from typing import Optional, List, Dict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

ROUTER_MODEL = os.getenv("ROUTER_MODEL", "llama3.1:8b")
router_llm = ChatOllama(model=ROUTER_MODEL, temperature=0)

# ---------------------------------------------------------------------------
# Model auto-selection
# ---------------------------------------------------------------------------
# Which local model handles each concrete node/engine in the graph. This is
# now actually consumed by graph.py's orchestrator_node and doc_read_node —
# in the first draft this map existed but nothing ever called it.
STEP_MODEL_MAP: Dict[str, str] = {
    "ocr": os.getenv("VISION_MODEL", "llava"),
    "vision": os.getenv("VISION_MODEL", "llava"),
    "rag": os.getenv("ROUTER_MODEL", "llama3.1:8b"),
    "coding_agent": os.getenv("CODING_MODEL", "qwen2.5-coder:7b"),
    "code_review_agent": os.getenv("CODING_MODEL", "qwen2.5-coder:7b"),
    "doc_write": os.getenv("ROUTER_MODEL", "llama3.1:8b"),
}
DEFAULT_MODEL = ROUTER_MODEL


def get_model_for_step(step_name: str) -> str:
    """Return the local model assigned to a given node/engine name."""
    return STEP_MODEL_MAP.get(step_name, DEFAULT_MODEL)


def build_step_models(steps: List[str]) -> Dict[str, str]:
    """Given a list of node/engine names, return {step: model}."""
    return {step: get_model_for_step(step) for step in steps}


# ---------------------------------------------------------------------------
# Tier 2: semantic router (used when Tier 1 regex in graph.py is ambiguous)
# ---------------------------------------------------------------------------
PLANNER_SYSTEM_PROMPT = """
You are the central Orchestrator Brain for a Sovereign AI system.
Classify the user task into exactly ONE operational route:
- "coding": Calculation, script execution, numerical verification, math checks.
- "doc_read": Needs OCR/text extraction, visual diagram reading, or SOP/manual search.
- "doc_write": Direct report/note generation with no extraction step needed first.
- "clarification": Unclear, nonsensical (e.g. random keyboard mashing), or unmappable.

Rules:
1. If the prompt is nonsensical or cannot be mapped to a task, return {"route": "clarification"}.
2. Otherwise return ONLY valid JSON matching this schema, nothing else:
{"route": "coding" | "doc_read" | "doc_write" | "clarification"}
No preamble, no markdown fences, no explanation.
"""


def level_2_llm_router(prompt: str, has_file: bool) -> str:
    """Semantic fallback classifier for prompts the Tier-1 regex can't resolve."""
    try:
        response = router_llm.invoke([
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"User Task: {prompt}\nFile Attached: {has_file}")
        ])
        content = str(response.content).strip()
        if "```" in content:
            content = content.split("```json")[-1].split("```")[0].strip()
        data = json.loads(content)
        route = data.get("route")
        if route not in ("coding", "doc_read", "doc_write", "clarification"):
            return "clarification"
        return route
    except Exception:
        # Fail loud: an unrouteable request should ask for clarification,
        # not silently masquerade as a doc_write task.
        return "clarification"


# ---------------------------------------------------------------------------
# Step decomposition — this is the piece that was previously dead code
# ---------------------------------------------------------------------------
def build_execution_plan(route: str, prompt: str) -> List[str]:
    """
    Turn a single route into an ordered list of graph stages. Every
    non-clarification route ends in doc_write, because doc_write is the
    mandatory final composition stage before validation/guardrails/artifact
    generation in this architecture (see the pipeline diagram).
    """
    if route == "clarification":
        return []

    prompt_l = (prompt or "").lower()
    wants_coding = any(k in prompt_l for k in (
        "code", "calculate", "calculation", "math", "verify", "script",
        "compute", "factor", "equation", "check the numbers"
    ))

    if route == "coding":
        return ["coding_agent", "doc_write"]

    if route == "doc_read":
        plan = ["doc_read"]
        if wants_coding:
            plan.append("coding_agent")
        plan.append("doc_write")
        return plan

    # route == "doc_write" or any unexpected value
    return ["doc_write"]


def determine_sub_engines(prompt: str, file_path: Optional[str]) -> List[str]:
    """
    Decide which ingestion engine(s) doc_read needs to run, in order.
    A single request can legitimately need more than one engine (e.g. a
    scanned inspection sheet that also needs an SOP lookup) — the first
    draft's doc_read could only ever branch to exactly one engine.
    """
    prompt_l = (prompt or "").lower()
    path_l = (file_path or "").lower()
    has_file = bool(file_path and file_path.strip())

    is_image = path_l.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"))
    wants_vision = is_image or any(
        k in prompt_l for k in ("image", "drawing", "diagram", "schematic", "photograph", "photo")
    )
    wants_rag = any(k in prompt_l for k in ("sop", "standard", "manual", "clause", "policy", "regulation"))
    explicit_scan = any(k in prompt_l for k in ("scan", "ocr", "handwritten"))
    wants_ocr = (has_file and not is_image) or explicit_scan

    engines: List[str] = []
    if wants_vision:
        engines.append("vision")
    if wants_ocr and "ocr" not in engines:
        engines.append("ocr")
    if wants_rag:
        engines.append("rag")

    if not engines:
        # Something triggered doc_read (a file, or a "read/extract" style
        # prompt) but nothing more specific matched — default to OCR/text
        # extraction rather than silently doing nothing.
        engines.append("ocr")

    return engines