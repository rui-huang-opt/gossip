import numpy as np
from numpy.typing import NDArray
from abc import ABCMeta, abstractmethod
from typing import List, KeysView, TypeVar, Generic, Dict

T = TypeVar("T")


class Gossip(Generic[T], metaclass=ABCMeta):
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
        self._connections: Dict[str, T] = {}

    @property
    def degree(self) -> int:
        return len(self._connections)

    @property
    def neighbor_names(self) -> KeysView[str]:
        return self._connections.keys()

    @abstractmethod
    def send(self, name: str, state: NDArray[np.float64], *args, **kwargs): ...

    @abstractmethod
    def recv(self, name: str, *args, **kwargs) -> NDArray[np.float64] | None: ...

    @abstractmethod
    def _broadcast_with_noise(self, state: NDArray[np.float64], *args, **kwargs): ...

    @abstractmethod
    def _broadcast_without_noise(self, state: NDArray[np.float64], *args, **kwargs): ...

    def broadcast(self, state: NDArray[np.float64], *args, **kwargs):
        if self._noise_scale:
            self._broadcast_with_noise(state, *args, **kwargs)
        else:
            self._broadcast_without_noise(state, *args, **kwargs)

    @abstractmethod
    def gather(self, *args, **kwargs) -> List[NDArray[np.float64]]: ...

    def add_connection(self, neighbor: str, conn: T):
        if neighbor in self._connections:
            raise ValueError(f"Connection to neighbor '{neighbor}' already exists")
        self._connections[neighbor] = conn

    def remove_connection(self, neighbor: str):
        if neighbor not in self._connections:
            raise ValueError(f"No connection to neighbor '{neighbor}'")
        self._connections.pop(neighbor)

    @abstractmethod
    def close(self): ...

    def compute_laplacian(
        self, state: NDArray[np.float64], *args, **kwargs
    ) -> NDArray[np.float64]:
        self.broadcast(state, *args, **kwargs)
        neighbor_states = self.gather(*args, **kwargs)

        return len(neighbor_states) * state - sum(neighbor_states)
