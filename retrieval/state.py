from dataclasses import dataclass, field


@dataclass
class AgentState:
    original_query: str
    current_strategy: str = "Expansion"
    retries_remaining: int = 3
    processed_queries: list = field(default_factory=list)
    # All query-sets issued so far — lets the dispatcher avoid repeating them.
    tried_queries: list = field(default_factory=list)
    reasoning: str = ""
    critique_log: list = field(default_factory=list)
    best_results: list = field(default_factory=list)
    # Results merged across all retries so partial hits aren't thrown away.
    accumulated_results: list = field(default_factory=list)
    # Descriptions generated from user-supplied images (set before the dispatch loop).
    image_descriptions: list = field(default_factory=list)
