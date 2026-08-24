from langgraph.graph import StateGraph, START, END
from state import AgentState
from planner import planner_node

def route_next_step(state: AgentState) -> str:
    plan = state.get("plan", [])
    index = state.get("current_step_index", 0)
    if index < len(plan):
        return plan[index]
    return "done"

# Default worker mocks (Teammates replace these later)
def doc_read_node(state: AgentState):
    print(f"--> [Step {state['current_step_index'] + 1}] Running Document Reader / Vision...")
    return {
        "extracted_data": "Pipe Wall Thickness: 12.4mm, Measured Pressure: 150 PSI",
        "current_step_index": state["current_step_index"] + 1
    }

def rag_search_node(state: AgentState):
    print(f"--> [Step {state['current_step_index'] + 1}] Querying SOP Knowledge Base...")
    return {
        "sop_rules": "SOP-04: Minimum allowed thickness is 10.0mm at 150 PSI.",
        "current_step_index": state["current_step_index"] + 1
    }

def coding_node(state: AgentState):
    print(f"--> [Step {state['current_step_index'] + 1}] Executing Math Verification in Sandbox...")
    return {
        "code_results": "Calculated Safety Factor: 1.42 (Compliant)",
        "current_step_index": state["current_step_index"] + 1
    }

def doc_write_node(state: AgentState):
    print(f"--> [Step {state['current_step_index'] + 1}] Generating Final Word Approval Note...")
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

def build_app(doc_fn=doc_read_node, rag_fn=rag_search_node, code_fn=coding_node, write_fn=doc_write_node):
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("doc_read", doc_fn)
    workflow.add_node("rag_search", rag_fn)
    workflow.add_node("coding", code_fn)
    workflow.add_node("doc_write", write_fn)

    workflow.add_edge(START, "planner")

    for node in ["planner", "doc_read", "rag_search", "coding", "doc_write"]:
        workflow.add_conditional_edges(
            node,
            route_next_step,
            {
                "doc_read": "doc_read",
                "rag_search": "rag_search",
                "coding": "coding",
                "doc_write": "doc_write",
                "done": END
            }
        )

    return workflow.compile()

if __name__ == "__main__":
    app = build_app()
    test_task = {
        "user_prompt": "Read the inspection PDF, verify against SOP rules, run safety calculations, and create the approval note",
        "plan": [],
        "current_step_index": 0
    }
    result = app.invoke(test_task)
    print("\n--- Execution Complete ---")
    print(result["generated_artifact"])