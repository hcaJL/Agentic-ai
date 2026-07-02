"""LangGraph version of the pipeline. Use this once you're past the initial
sequential version. `pip install langgraph` first.

The nodes are framework-agnostic (state in, state out); here we bind `engine` and
`persona` via closures and let LangGraph own the routing.
"""
from functools import partial


def build_graph(engine, persona: str = "calm"):
    from langgraph.graph import StateGraph, END
    from graph.state import BroadcastState
    from graph.nodes.perception import perception_node
    from graph.nodes.event_detector import event_detector_node
    from graph.nodes.director import director_node, route_selector
    from graph.nodes.memory import retrieve_memory_node, update_memory_node
    from graph.nodes.booth import light_commentary_node, booth_node

    g = StateGraph(BroadcastState)
    g.add_node("perception", partial(perception_node, engine=engine))
    g.add_node("detect_event", event_detector_node)
    g.add_node("director", director_node)
    g.add_node("light", partial(light_commentary_node, persona=persona))
    g.add_node("retrieve_memory", retrieve_memory_node)
    g.add_node("booth", partial(booth_node, persona=persona))
    g.add_node("update_memory", update_memory_node)

    g.set_entry_point("perception")
    g.add_edge("perception", "detect_event")
    g.add_edge("detect_event", "director")
    g.add_conditional_edges("director", route_selector,
                            {"light": "light", "deep": "retrieve_memory"})
    g.add_edge("retrieve_memory", "booth")
    g.add_edge("light", "update_memory")
    g.add_edge("booth", "update_memory")
    g.add_edge("update_memory", END)
    return g.compile()
