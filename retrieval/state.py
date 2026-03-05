from dataclasses import dataclass, field


@dataclass
class AgentState:
    original_query: str
    current_strategy: str = "Expansion"
    retries_remaining: int = 3
    processed_queries: list = field(default_factory=list)
    reasoning: str = ""
    critique_log: list = field(default_factory=list)
    best_results: list = field(default_factory=list)
