from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# 1. Full Shared State
class AgentState(TypedDict):
    user_prompt: str
    selected_route: Optional[str]
    generated_code: Optional[str]
    test_status: Literal["pass", "fail", "not_run"]
    retry_count: int
    error_log: Optional[str]
    final_output: Optional[str]

# 2. Router Schema
class RouteDecision(BaseModel):
    decision: str = Field(
        description="Must be 'coding', 'doc_read', or 'doc_write'."
    )

# 3. Nodes Definition
def orchestrator_node(state: AgentState):
    llm = ChatOllama(model="llama3.1:8b", format="json", temperature=0)
    prompt = f"Classify this request: '{state['user_prompt']}'. Options: 'coding', 'doc_read', or 'doc_write'."
    response = llm.with_structured_output(RouteDecision).invoke(prompt)
    print(f"\n[Orchestrator] --> Route: {response.decision}")
    return {"selected_route": response.decision, "retry_count": 0}

def coding_agent(state: AgentState):
    retries = state.get("retry_count", 0)
    print(f"[Coding Agent] Writing code (Attempt #{retries + 1})...")
    
    # Simulate a bug on attempt 1, but fix it on attempt 2
    if retries == 0:
        code = "print('Calculated Pressure: ' + 120) # intentional bug: string + int"
    else:
        code = "print('Calculated Pressure: ' + str(120)) # bug fixed"
        
    return {"generated_code": code}

def code_test_agent(state: AgentState):
    code = state["generated_code"]
    print(f"[Code Test] Running code in sandbox...")
    
    try:
        # Execute in isolated local environment
        exec_globals = {}
        exec(code, exec_globals)
        print("  -> Sandbox Result: Test PASSED")
        return {"test_status": "pass", "error_log": None}
    except Exception as e:
        print(f"  -> Sandbox Result: Test FAILED with error: {e}")
        return {
            "test_status": "fail", 
            "error_log": str(e),
            "retry_count": state["retry_count"] + 1
        }

def code_review_agent(state: AgentState):
    print("[Code Review] Reviewing code quality & security... Approved.")
    return {"final_output": f"Verified Code Ready: {state['generated_code']}"}

def doc_read_worker(state: AgentState):
    print("[Doc Read] Parsing document...")
    return {"final_output": "Document parsed successfully."}

def doc_write_worker(state: AgentState):
    print("[Doc Write] Generating draft...")
    return {"final_output": "Draft note created."}

# 4. Conditional Edge Functions
def route_initial_decision(state: AgentState):
    return state["selected_route"]

def route_test_results(state: AgentState):
    if state["test_status"] == "pass":
        return "pass"
    return "fail"

# 5. Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("coding_agent", coding_agent)
workflow.add_node("code_test_agent", code_test_agent)
workflow.add_node("code_review_agent", code_review_agent)
workflow.add_node("doc_read", doc_read_worker)
workflow.add_node("doc_write", doc_write_worker)

workflow.add_edge(START, "orchestrator")
workflow.add_conditional_edges(
    "orchestrator",
    route_initial_decision,
    {
        "coding": "coding_agent",
        "doc_read": "doc_read",
        "doc_write": "doc_write"
    }
)

# Coding feedback loop
workflow.add_edge("coding_agent", "code_test_agent")
workflow.add_conditional_edges(
    "code_test_agent",
    route_test_results,
    {
        "pass": "code_review_agent",
        "fail": "coding_agent"   # Auto-retry loop on failure
    }
)

workflow.add_edge("code_review_agent", END)
workflow.add_edge("doc_read", END)
workflow.add_edge("doc_write", END)

app = workflow.compile()

# 6. Test with a coding prompt
if __name__ == "__main__":
    prompt = {"user_prompt": "Generate a python script to calculate boiler pipe pressure"}
    result = app.invoke(prompt)
    print("\n--- Final Workflow Result ---")
    print(f"Final Output: {result['final_output']}")
    print(f"Total Retries Needed: {result['retry_count']}")