import numpy as np
from numpy.typing import NDArray
from numpy.random import normal
from typing import List, Tuple, Dict, KeysView
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from .gossip import Gossip


class SyncGossip(Gossip):
    """
    Synchronous gossip communication using multiprocessing pipes.
    This class implements the Gossip interface for synchronous communication.
    """

    def __init__(
        self,
        name: str,
        noise_scale: int | float | None = None,
    ):
        super().__init__(name, noise_scale)
        self._connections: Dict[str, Connection] = {}

    @property
    def degree(self) -> int:
        return len(self._connections)

    @property
    def neighbor_names(self) -> KeysView[str]:
        return self._connections.keys()

    def _broadcast_with_noise(self, state: NDArray[np.float64], index: int = 0):
        for conn in self._connections.values():
            noise = normal(scale=self._noise_scale, size=state.shape)
            conn.send(state + noise)

    def _broadcast_without_noise(self, state: NDArray[np.float64], index: int = 0):
        for conn in self._connections.values():
            conn.send(state)

    def gather(self, index: int = 0) -> List[NDArray[np.float64]]:
        return [conn.recv() for conn in self._connections.values()]

    def add_connection(self, name: str, conn: Connection):
        if name in self._connections:
            return
        self._connections[name] = conn

    def remove_connection(self, name: str):
        self._connections.pop(name, None)


def create_sync_network(
    node_names: List[str],
    edge_pairs: List[Tuple[str, str]],
    noise_scale: int | float | None = None,
) -> Dict[str, Gossip]:
    """
    Create a synchronous gossip network with the given node names and edge pairs.
    Each node is represented by a SyncGossip instance, and connections are established
    using multiprocessing pipes.

    Args:
        node_names (List[str]): List of node names.
        edge_pairs (List[Tuple[str, str]]): List of tuples representing edges between nodes.
        noise_scale (int | float | None): Scale of noise to be added to the state.
    Returns:
        Dict[str, Gossip]: Dictionary mapping node names to SyncGossip instances.
    """

    gossip_map = {name: SyncGossip(name, noise_scale) for name in node_names}

    for u, v in edge_pairs:
        conn_u, conn_v = Pipe()
        gossip_map[u].add_connection(v, conn_u)
        gossip_map[v].add_connection(u, conn_v)

    return gossip_map
