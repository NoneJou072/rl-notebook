
class ContinuesBase:
    def __init__(self, args) -> None:
        self.memory = None
        self.agent_name = None

    def sample_action(self, s, deterministic=False):
        raise NotImplementedError

    def update(self, *args):
        raise NotImplementedError
