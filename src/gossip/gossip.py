import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Tuple
from multiprocessing import Pipe
from multiprocessing.connection import Connection


class Gossip:
    """
    A class to manage the gossip protocol for a node in a distributed network.

    This class manages communication between nodes in a gossip network.
    It allows sending and receiving messages to/from neighbors, broadcasting
    messages to all neighbors, and gathering messages from all neighbors.
    """

    def __init__(self, connections: Dict[str, Connection] = None):
        self._connections = connections if connections is not None else {}

    @property
    def degree(self) -> int:
        return len(self._connections)

    @property
    def neighbor_ids(self) -> List[str]:
        return self._connections.keys()

    def send(self, id: str, state: NDArray[np.float64]):
        self._connections[id].send(state)

    def recv(self, id: str) -> NDArray[np.float64]:
        return self._connections[id].recv()

    def broadcast(self, state: NDArray[np.float64]):
        for conn in self._connections.values():
            conn.send(state)

    def gather(self) -> List[NDArray[np.float64]]:
        return [conn.recv() for conn in self._connections.values()]

    def add_connection(self, neighbor: str, conn: Connection):
        self._connections[neighbor] = conn

    def remove_connection(self, neighbor: str):
        self._connections.pop(neighbor)

    def close(self):
        for conn in self._connections.values():
            conn.close()


def create_gossip_network(
    nodes: List[str], edges: List[Tuple[str, str]]
) -> Dict[str, Gossip]:
    """
    Create a gossip network from a list of nodes and edges.
    """

    gossips = {node: Gossip() for node in nodes}

    for edge in edges:
        conn = Pipe()
        gossips[edge[0]].add_connection(edge[1], conn[0])
        gossips[edge[1]].add_connection(edge[0], conn[1])

    return gossips
