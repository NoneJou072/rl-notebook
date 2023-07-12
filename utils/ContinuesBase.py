
class ContinuesBase:
    def __init__(self, cfg) -> None:
        self.memory = None

    def sample_action(self, s, deterministic=False):
        raise NotImplementedError

    def update(self, *args):
        raise NotImplementedError
