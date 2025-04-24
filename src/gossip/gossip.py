import numpy as np
from numpy.typing import NDArray
from numpy.random import normal
from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Dict
from multiprocessing import Pipe
from multiprocessing.connection import Connection


class Gossip(metaclass=ABCMeta):
    """
    A class to manage the gossip protocol for a node in a distributed network.

    This class manages communication between nodes in a gossip network.
    It allows sending and receiving messages to/from neighbors, broadcasting
    messages to all neighbors, and gathering messages from all neighbors.
    """

    def __init__(
        self,
        name: str,
        noise_scale: int | float | None,
    ):
        self.name = name
        self._noise_scale = noise_scale
        self._connections: Dict[str, Connection] = {}

    @property
    def degree(self) -> int:
        return len(self._connections)

    @property
    def neighbor_names(self) -> List[str]:
        return self._connections.keys()

    @abstractmethod
    def send(self, name: str, state: NDArray[np.float64]): ...

    @abstractmethod
    def recv(self, name: str) -> NDArray[np.float64]: ...

    @abstractmethod
    def broadcast(self, state: NDArray[np.float64]): ...

    @abstractmethod
    def gather(self) -> List[NDArray[np.float64]]: ...

    def add_connection(self, neighbor: str, conn: Connection):
        self._connections[neighbor] = conn

    def remove_connection(self, neighbor: str):
        self._connections.pop(neighbor)

    @abstractmethod
    def close(self): ...

    @abstractmethod
    def compute_laplacian(self, state: NDArray[np.float64]) -> NDArray[np.float64]: ...


class SyncGossip(Gossip):
    """
    SyncGossip is a class that handles the synchronization of gossip messages
    between nodes in a distributed system. It ensures that all nodes have the
    same view of the system state by exchanging messages and updating their
    local state accordingly.
    """

    def __init__(self, name: str, noise_scale: int | float | None = None):
        super().__init__(name, noise_scale)

    def send(self, name: str, state: NDArray[np.float64]):
        if self._noise_scale is None:
            self._connections[name].send(state)
        else:
            noise = normal(scale=self._noise_scale, size=state.shape)
            self._connections[name].send(state + noise)

    def recv(self, name: str) -> NDArray[np.float64]:
        return self._connections[name].recv()

    def broadcast(self, state: NDArray[np.float64]):
        for conn in self._connections.values():
            if self._noise_scale is None:
                conn.send(state)
            else:
                noise = normal(scale=self._noise_scale, size=state.shape)
                conn.send(state + noise)

    def gather(self) -> List[NDArray[np.float64]]:
        return [conn.recv() for conn in self._connections.values()]

    def close(self):
        for conn in self._connections.values():
            conn.close()

    def compute_laplacian(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        self.broadcast(state)
        neighbor_states = self.gather()

        return self.degree * state - sum(neighbor_states)


class AsyncGossip(Gossip):
    pass


def create_gossip_network(
    node_names: List[str],
    edge_pairs: List[Tuple[str, str]],
    noise_scale: int | float | None = None,
    is_async: bool = False,
) -> Dict[str, Gossip]:
    """
    Create a gossip network from a list of nodes and edges.

    Parameters
    ----------
    node_names : List[str]
        A list of node identifiers.
    edge_pairs : List[Tuple[str, str]]
        A list of pairs of node identifiers representing edges in the network.

    Returns
    -------
    Dict[str, Gossip]
        A dictionary of gossip communicators indexed by node identifier.
    """

    gossip_map = {name: SyncGossip(name, noise_scale) for name in node_names}

    for u, v in edge_pairs:
        conn_u, conn_v = Pipe()
        gossip_map[u].add_connection(v, conn_u)
        gossip_map[v].add_connection(u, conn_v)

    return gossip_map
