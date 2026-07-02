"""Director — the agentic core. Decides route + emotional register per move.

route is what makes this NOT a linear pipeline: routine moves take the cheap path,
critical moves trigger deep analysis + memory + the full booth.
"""
import config


def director_node(state: dict) -> dict:
    ev = state["event"]
    label = ev["severity_label"]

    route = "deep" if label == "critical" else "light"

    # Pacing nudge: if we've been quiet/light for a while, promote a notable to deep
    # so the broadcast doesn't go flat.
    light_streak = state.get("_light_streak", 0)
    if route == "light" and label == "notable" and light_streak >= 4:
        route = "deep"

    state["route"] = route
    state["_light_streak"] = 0 if route == "deep" else light_streak + 1

    # Tone is DERIVED from severity — never a user dropdown.
    intensity = config.INTENSITY[label]
    state["register_intensity"] = intensity
    return state


def route_selector(state: dict) -> str:
    """For LangGraph add_conditional_edges."""
    return state["route"]
