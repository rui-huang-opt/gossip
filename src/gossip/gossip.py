import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Dict
from multiprocessing import Pipe
from multiprocessing.connection import Connection


class Gossip:
    """
    A class to manage the gossip protocol for a node in a distributed network.

    This class manages communication between nodes in a gossip network.
    It allows sending and receiving messages to/from neighbors, broadcasting
    messages to all neighbors, and gathering messages from all neighbors.
    """

    def __init__(self, id: str, **connections: Connection):
        self.id = id
        self._connections = connections

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
    node_ids: List[str], edge_pairs: List[Tuple[str, str]]
) -> Dict[str, Gossip]:
    """
    Create a gossip network from a list of nodes and edges.

    Parameters
    ----------
    node_ids : List[str]
        A list of node identifiers.
    edge_pairs : List[Tuple[str, str]]
        A list of pairs of node identifiers representing edges in the network.

    Returns
    -------
    Dict[str, Gossip]
        A dictionary of gossip communicators indexed by node identifier.
    """

    gossip_map = {id: Gossip(id) for id in node_ids}

    for node1, node2 in edge_pairs:
        conn1, conn2 = Pipe()
        gossip_map[node1].add_connection(node2, conn1)
        gossip_map[node2].add_connection(node1, conn2)

    return gossip_map
