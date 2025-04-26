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
        self.name = name
        self._noise_scale = noise_scale

    @property
    @abstractmethod
    def degree(self) -> int: ...

    @property
    @abstractmethod
    def neighbor_names(self) -> KeysView[str]: ...

    @abstractmethod
    def send(self, name: str, state: NDArray[np.float64]): ...

    @abstractmethod
    def recv(self, name: str) -> NDArray[np.float64]: ...

    @abstractmethod
    def _broadcast_with_noise(self, state: NDArray[np.float64]): ...

    @abstractmethod
    def _broadcast_without_noise(self, state: NDArray[np.float64]): ...

    def broadcast(self, state: NDArray[np.float64]):
        if self._noise_scale:
            self._broadcast_with_noise(state)
        else:
            self._broadcast_without_noise(state)

    @abstractmethod
    def gather(self) -> List[NDArray[np.float64]]: ...

    @abstractmethod
    def add_connection(self, neighbor: str, *args, **kwargs): ...

    @abstractmethod
    def remove_connection(self, neighbor: str): ...

    @abstractmethod
    def close(self): ...

    def compute_laplacian(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        self.broadcast(state)
        neighbor_states = self.gather()

        return len(neighbor_states) * state - sum(neighbor_states)
