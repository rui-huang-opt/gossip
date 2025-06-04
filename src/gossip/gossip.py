import numpy as np
from numpy.typing import NDArray
from abc import ABCMeta, abstractmethod
from typing import List, KeysView, Any


class Gossip(metaclass=ABCMeta):
    """
    Abstract base class for gossip communication.
    This class defines the interface for both synchronous and asynchronous gossip communication.
    """

    def __init__(
        self,
        name: Any,
        noise_scale: float | None,
    ):
        self._name = name
        self._noise_scale = noise_scale

    @property
    def name(self) -> Any:
        return self._name

    @property
    @abstractmethod
    def degree(self) -> int: ...

    @property
    @abstractmethod
    def neighbor_names(self) -> KeysView[Any]: ...

    @abstractmethod
    def send(self, name: Any, state: NDArray[np.float64], index: int = 0): ...

    @abstractmethod
    def recv(self, name: Any, index: int = 0) -> NDArray[np.float64]: ...

    @abstractmethod
    def broadcast(self, state: NDArray[np.float64], index: int = 0): ...

    @abstractmethod
    def gather(self, index: int = 0) -> List[NDArray[np.float64]]: ...

    def compute_laplacian(
        self, state: NDArray[np.float64], index: int = 0
    ) -> NDArray[np.float64]:
        self.broadcast(state, index)
        neighbor_states = self.gather(index)

        return len(neighbor_states) * state - sum(neighbor_states)
