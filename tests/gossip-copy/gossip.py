import numpy as np
from numpy.typing import NDArray
from abc import ABCMeta, abstractmethod
from typing import List, KeysView


class Gossip(metaclass=ABCMeta):
    """
    Abstract base class for gossip communication.
    This class defines the interface for both synchronous and asynchronous gossip communication.
    """

    def __init__(
        self,
        name: str,
        noise_scale: int | float | None,
    ):
        self._name = name
        self._noise_scale = noise_scale

    @property
    def name(self) -> str:
        return self._name

    @property
    @abstractmethod
    def degree(self) -> int: ...

    @property
    @abstractmethod
    def neighbor_names(self) -> List[str] | KeysView[str]: ...

    @abstractmethod
    def _broadcast_with_noise(self, state: NDArray[np.float64], index: int = 0): ...

    @abstractmethod
    def _broadcast_without_noise(self, state: NDArray[np.float64], index: int = 0): ...

    def broadcast(self, state: NDArray[np.float64], index: int = 0):
        if self._noise_scale:
            self._broadcast_with_noise(state, index)
        else:
            self._broadcast_without_noise(state, index)

    @abstractmethod
    def gather(self, index: int = 0) -> List[NDArray[np.float64]]: ...

    def init(self):
        pass

    def compute_laplacian(
        self, state: NDArray[np.float64], index: int = 0
    ) -> NDArray[np.float64]:
        self.broadcast(state, index)
        neighbor_states = self.gather(index)

        return len(neighbor_states) * state - sum(neighbor_states)
