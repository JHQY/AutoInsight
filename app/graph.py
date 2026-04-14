# app/graph.py
from langgraph.graph import StateGraph, END
from app.state import AgentState
from nodes.profiling      import profiling_node
from nodes.intent_routing import intent_routing_node
from nodes.processing     import data_processing
from nodes.eda            import run_eda
from nodes.model_routing  import model_routing_node
from nodes.modeling       import modeling_node
from nodes.evaluation     import evaluation_node
from nodes.inference      import inference_node
from nodes.reporting      import generate_report


def _route_after_model_routing(state: AgentState) -> str:
    """correlation_analysis skips Modeling + Evaluation + Inference, goes directly to Reporting."""
    if state.get("task_type") == "correlation_analysis":
        return "reporting"
    return "modeling"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("profiling",      profiling_node)
    graph.add_node("intent_routing", intent_routing_node)
    graph.add_node("processing",     data_processing)
    graph.add_node("eda",            run_eda)
    graph.add_node("model_routing",  model_routing_node)
    graph.add_node("modeling",       modeling_node)
    graph.add_node("evaluation",     evaluation_node)
    graph.add_node("inference",      inference_node)
    graph.add_node("reporting",      generate_report)

    graph.set_entry_point("profiling")
    graph.add_edge("profiling",      "intent_routing")
    graph.add_edge("intent_routing", "processing")
    graph.add_edge("processing",     "eda")
    graph.add_edge("eda",            "model_routing")
    graph.add_conditional_edges(
        "model_routing",
        _route_after_model_routing,
        {"modeling": "modeling", "reporting": "reporting"},
    )
    graph.add_edge("modeling",       "evaluation")
    graph.add_edge("evaluation",     "inference")
    graph.add_edge("inference",      "reporting")
    graph.add_edge("reporting",      END)

    return graph.compile()
