import numpy as np
from numpy.typing import NDArray
from numpy.random import normal
from typing import KeysView, Mapping
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from ..gossip import Gossip


class SyncGossip(Gossip):
    """
    Synchronous gossip communication using multiprocessing pipes.
    This class implements the Gossip interface for synchronous communication.
    """

    def __init__(
        self,
        name: str,
        noise_scale: float | None = None,
    ):
        super().__init__(name, noise_scale)
        self._connections: dict[str, Connection] = {}

    @property
    def degree(self) -> int:
        return len(self._connections)

    @property
    def neighbor_names(self) -> KeysView[str]:
        return self._connections.keys()

    def broadcast(self, state: NDArray[np.float64], index: int = 0):
        if self._noise_scale is None:
            for conn in self._connections.values():
                conn.send(state)
        else:
            for conn in self._connections.values():
                noise = normal(scale=self._noise_scale, size=state.shape)
                conn.send(state + noise)

    def gather(self, index: int = 0) -> list[NDArray[np.float64]]:
        return [conn.recv() for conn in self._connections.values()]

    def add_connection(self, name: str, conn: Connection):
        if name in self._connections:
            return
        self._connections[name] = conn

    def remove_connection(self, name: str):
        self._connections.pop(name, None)


def create_sync_network(
    node_names: list[str],
    edge_pairs: list[tuple[str, str]],
    noise_scale: float | None = None,
) -> Mapping[str, Gossip]:
    """
    Create a synchronous gossip network with the given node names and edge pairs.
    Each node is represented by a SyncGossip instance, and connections are established
    using multiprocessing pipes.

    Args:
        node_names (List[str]): List of node names.
        edge_pairs (List[Tuple[str, str]]): List of tuples representing edges between nodes.
        noise_scale (float | None): Scale of noise to be added to the state.
    Returns:
        Dict[str, Gossip]: Dictionary mapping node names to SyncGossip instances.
    """

    network = {name: SyncGossip(name, noise_scale) for name in node_names}

    for u, v in edge_pairs:
        conn_u, conn_v = Pipe()
        network[u].add_connection(v, conn_u)
        network[v].add_connection(u, conn_v)

    return network
