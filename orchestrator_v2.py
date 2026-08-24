from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# 1. State Schema
class AgentState(TypedDict):
    user_prompt: str
    plan: List[str]
    current_step_index: int
    extracted_data: Optional[str]
    sop_rules: Optional[str]
    code_results: Optional[str]
    generated_artifact: Optional[str]

# 2. Structured Output Schema for the Planner
class PlanOutput(BaseModel):
    steps: List[str] = Field(
        description="Ordered list of steps. Allowed values: 'doc_read', 'rag_search', 'coding', 'doc_write'."
    )

# 3. Planner Node
def planner_node(state: AgentState):
    llm = ChatOllama(model="llama3.1:8b", format="json", temperature=0)
    prompt = f"""
    You are an industrial orchestrator. Break down this task into ordered steps.
    Task: "{state['user_prompt']}"
    
    Allowed steps: ['doc_read', 'rag_search', 'coding', 'doc_write']
    """
    plan_result = llm.with_structured_output(PlanOutput).invoke(prompt)
    print(f"\n[Planner] Generated execution plan: {plan_result.steps}")
    return {"plan": plan_result.steps, "current_step_index": 0}

# 4. Worker Nodes
def doc_read_node(state: AgentState):
    print(f"--> [Step {state['current_step_index'] + 1}] Reading and extracting document data...")
    return {
        "extracted_data": "Pipe Wall Thickness: 12.4mm, Measured Pressure: 150 PSI",
        "current_step_index": state["current_step_index"] + 1
    }

def rag_search_node(state: AgentState):
    print(f"--> [Step {state['current_step_index'] + 1}] Searching SOP database for compliance limits...")
    return {
        "sop_rules": "SOP-04: Minimum allowed thickness is 10.0mm at 150 PSI.",
        "current_step_index": state["current_step_index"] + 1
    }

def coding_node(state: AgentState):
    print(f"--> [Step {state['current_step_index'] + 1}] Executing mathematical stress verification in sandbox...")
    return {
        "code_results": "Calculated Safety Factor: 1.42 (Compliant)",
        "current_step_index": state["current_step_index"] + 1
    }

def doc_write_node(state: AgentState):
    print(f"--> [Step {state['current_step_index'] + 1}] Generating final Word approval memo...")
    summary = (
        f"Deliverable Summary:\n"
        f" - Findings: {state.get('extracted_data')}\n"
        f" - Standards: {state.get('sop_rules')}\n"
        f" - Math Verification: {state.get('code_results', 'N/A')}"
    )
    return {
        "generated_artifact": summary,
        "current_step_index": state["current_step_index"] + 1
    }

# 5. Dynamic Step Dispatcher
def route_next_step(state: AgentState):
    plan = state["plan"]
    index = state["current_step_index"]
    
    if index < len(plan):
        return plan[index]
    return "done"

# 6. Assemble the Graph
workflow = StateGraph(AgentState)

# Register ALL nodes
workflow.add_node("planner", planner_node)
workflow.add_node("doc_read", doc_read_node)
workflow.add_node("rag_search", rag_search_node)
workflow.add_node("coding", coding_node)
workflow.add_node("doc_write", doc_write_node)

workflow.add_edge(START, "planner")

# Conditional edges for sequential routing
all_nodes = ["planner", "doc_read", "rag_search", "coding", "doc_write"]
for node_name in all_nodes:
    workflow.add_conditional_edges(
        node_name,
        route_next_step,
        {
            "doc_read": "doc_read",
            "rag_search": "rag_search",
            "coding": "coding",
            "doc_write": "doc_write",
            "done": END
        }
    )

app = workflow.compile()

# 7. Run Test
if __name__ == "__main__":
    prompt = {"user_prompt": "Read the inspection PDF, verify against SOP rules, run safety calculations, and create the approval note"}
    result = app.invoke(prompt)
    print("\n--- Final Deliverable ---")
    print(result["generated_artifact"])