import numpy as np
from numpy.typing import NDArray
from numpy.random import normal
from typing import List, Tuple, Dict, KeysView, NamedTuple
from multiprocessing import Queue
from .gossip import Gossip


class Channel(NamedTuple):
    """
    A named tuple representing a communication channel.
    It contains an input queue and an output queue for asynchronous communication.
    """

    in_queue: Queue
    out_queue: Queue


class AsyncGossip(Gossip):
    """
    Asynchronous gossip communication using multiprocessing queues.
    This class implements the Gossip interface for asynchronous communication.
    """

    def __init__(self, name: str, noise_scale: int | float | None = None):
        super().__init__(name, noise_scale)
        self._channels: Dict[str, Channel] = {}

    @property
    def degree(self) -> int:
        return len(self._channels)

    @property
    def neighbor_names(self) -> KeysView[str]:
        return self._channels.keys()

    def send(self, name: str, state: NDArray[np.float64]):
        channel = self._channels.get(name)
        if channel is None:
            raise ValueError(f"No connection to neighbor '{name}'")
        if channel.out_queue.full():
            channel.out_queue.get()
        if self._noise_scale:
            noise = normal(scale=self._noise_scale, size=state.shape)
            channel.out_queue.put(state + noise)
        else:
            channel.out_queue.put(state)

    def recv(self, name: str) -> NDArray[np.float64] | None:
        channel = self._channels.get(name)
        if channel is None:
            raise ValueError(f"No connection to neighbor '{name}'")
        return channel.in_queue.get() if not channel.in_queue.empty() else None

    def _broadcast_with_noise(self, state: NDArray[np.float64]):
        for channel in self._channels.values():
            if channel.out_queue.full():
                channel.out_queue.get()
            noise = normal(scale=self._noise_scale, size=state.shape)
            channel.out_queue.put(state + noise)

    def _broadcast_without_noise(self, state: NDArray[np.float64]):
        for channel in self._channels.values():
            if channel.out_queue.full():
                channel.out_queue.get()
            channel.out_queue.put(state)

    def gather(self) -> List[NDArray[np.float64]]:
        return [
            channel.in_queue.get() for channel in self._channels.values() if not channel.in_queue.empty()
        ]

    def add_connection(self, neighbor: str, in_queue: Queue, out_queue: Queue):
        if neighbor in self._channels:
            raise ValueError(f"Connection to neighbor '{neighbor}' already exists")
        self._channels[neighbor] = Channel(in_queue, out_queue)

    def remove_connection(self, neighbor: str):
        if neighbor not in self._channels:
            raise ValueError(f"No connection to neighbor '{neighbor}'")
        self._channels.pop(neighbor)

    def close(self):
        for channel in self._channels.values():
            while not channel.in_queue.empty():
                channel.in_queue.get()
            while not channel.out_queue.empty():
                channel.out_queue.get()
            channel.in_queue.close()
            channel.out_queue.close()


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
        gossip_map[u].add_connection(v, queue_u, queue_v)
        gossip_map[v].add_connection(u, queue_v, queue_u)

    return gossip_map
