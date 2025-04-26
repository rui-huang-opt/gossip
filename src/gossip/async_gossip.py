import numpy as np
from numpy.typing import NDArray
from numpy.random import normal
from typing import List, Tuple, Dict, NamedTuple
from multiprocessing import Queue
from .gossip import Gossip


class Connection(NamedTuple):
    """
    A named tuple representing a communication connection.
    It contains an input queue and an output queue for asynchronous communication.
    """

    in_queue: Queue
    out_queue: Queue


class AsyncGossip(Gossip[Connection]):
    """
    Asynchronous gossip communication using multiprocessing queues.
    This class implements the Gossip interface for asynchronous communication.
    """

    def __init__(self, name: str, noise_scale: int | float | None = None):
        super().__init__(name, noise_scale)

    def send(self, name: str, state: NDArray[np.float64], *args, **kwargs):
        connection = self._connections.get(name)
        if connection is None:
            raise ValueError(f"No connection to neighbor '{name}'")
        if connection.out_queue.full():
            connection.out_queue.get()
        if self._noise_scale:
            noise = normal(scale=self._noise_scale, size=state.shape)
            connection.out_queue.put(state + noise)
        else:
            connection.out_queue.put(state)

    def recv(self, name: str, *args, **kwargs) -> NDArray[np.float64] | None:
        connection = self._connections.get(name)
        if connection is None:
            raise ValueError(f"No connection to neighbor '{name}'")
        return connection.in_queue.get() if not connection.in_queue.empty() else None

    def _broadcast_with_noise(self, state: NDArray[np.float64], *args, **kwargs):
        for connection in self._connections.values():
            if connection.out_queue.full():
                connection.out_queue.get()
            noise = normal(scale=self._noise_scale, size=state.shape)
            connection.out_queue.put(state + noise)

    def _broadcast_without_noise(self, state: NDArray[np.float64], *args, **kwargs):
        for connection in self._connections.values():
            if connection.out_queue.full():
                connection.out_queue.get()
            connection.out_queue.put(state)

    def gather(self, *args, **kwargs) -> List[NDArray[np.float64]]:
        return [
            connection.in_queue.get()
            for connection in self._connections.values()
            if not connection.in_queue.empty()
        ]

    def close(self):
        for connection in self._connections.values():
            while not connection.in_queue.empty():
                connection.in_queue.get()
            while not connection.out_queue.empty():
                connection.out_queue.get()
            connection.in_queue.close()
            connection.out_queue.close()


def create_async_network(
    node_names: List[str],
    edge_pairs: List[Tuple[str, str]],
    noise_scale: int | float | None = None,
    maxsize: int = 10,
) -> Dict[str, Gossip]:
    """
    Create a network of gossip nodes with asynchronous communication.
    Each node is represented by a name and can communicate with its neighbors
    through a pair of queues (input and output).
    Args:
        node_names (List[str]): List of node names.
        edge_pairs (List[Tuple[str, str]]): List of edges represented as tuples of node names.
        noise_scale (int | float | None): Standard deviation of the noise to be added to the state.
        maxsize (int): Maximum size of the queues.
    Returns:
        Dict[str, Gossip]: A dictionary mapping node names to their corresponding AsyncGossip instances.
    """

    gossip_map = {name: AsyncGossip(name, noise_scale) for name in node_names}

    for u, v in edge_pairs:
        queue_u, queue_v = Queue(maxsize=maxsize), Queue(maxsize=maxsize)
        conn_u = Connection(queue_u, queue_v)
        conn_v = Connection(queue_v, queue_u)
        gossip_map[u].add_connection(v, conn_u)
        gossip_map[v].add_connection(u, conn_v)

    return gossip_map
