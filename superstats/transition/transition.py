from collections.abc import Callable
from abc import ABC, abstractmethod


class Transition(ABC):

    def __init__(self, initial_prior: Callable):
        pass

    def sample(self, steps: int):
        pass
