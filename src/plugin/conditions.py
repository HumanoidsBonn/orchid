"""Fast, read-only registered predicates; expensive checks belong in actions."""

class WorldStateExists:
    """Check whether a named world-state value is present (including false or zero)."""
    def __init__(self, key: str):
        self.key = key

    def __call__(self, context) -> bool:
        return context.world_state is not None and context.world_state.get(self.key) is not None
