import numpy as np
from numpy.typing import NDArray
from numpy.random import normal
from typing import List, Tuple, Dict, NamedTuple
from multiprocessing import Queue
from .gossip import Gossip


class Channel(NamedTuple):
    """
    A named tuple representing a communication channel.
    It contains an input and output queue for asynchronous communication.
    """

    in_queue: Queue
    out_queue: Queue


class Connection:
    """
    A class representing a connection between two nodes in the gossip network.
    It contains two queues for input and output communication.
    """

    def __init__(self, *channels: Channel):
        self._channels: Tuple[Channel, ...] = channels

    def full(self, index: int) -> bool:
        return self._channels[index].out_queue.full()

    def empty(self, index: int) -> bool:
        return self._channels[index].in_queue.empty()

    def send(self, state: NDArray[np.float64], index: int):
        if self._channels[index].out_queue.full():
            self._channels[index].out_queue.get()
        self._channels[index].out_queue.put(state)

    def recv(self, index: int) -> NDArray[np.float64]:
        return self._channels[index].in_queue.get()

    def close(self):
        for channel in self._channels:
            channel.in_queue.close()
            channel.out_queue.close()


class AsyncGossip(Gossip[Connection]):
    """
    Asynchronous gossip communication using multiprocessing queues.
    This class implements the Gossip interface for asynchronous communication.
    """

    def __init__(self, name: str, noise_scale: int | float | None = None):
        super().__init__(name, noise_scale)

    def send(self, name: str, state: NDArray[np.float64], index: int = 0):
        connection = self._connections.get(name)
        if connection is None:
            raise ValueError(f"No connection to neighbor '{name}'")
        if self._noise_scale:
            noise = normal(scale=self._noise_scale, size=state.shape)
            connection.send(state + noise, index)
        else:
            connection.send(state, index)

    def recv(self, name: str, index: int = 0) -> NDArray[np.float64] | None:
        connection = self._connections.get(name)
        if connection is None:
            raise ValueError(f"No connection to neighbor '{name}'")
        return connection.recv(index) if not connection.empty(index) else None

    def _broadcast_with_noise(self, state: NDArray[np.float64], index: int = 0):
        for connection in self._connections.values():
            noise = normal(scale=self._noise_scale, size=state.shape)
            connection.send(state + noise, index)

    def _broadcast_without_noise(self, state: NDArray[np.float64], index: int = 0):
        for connection in self._connections.values():
            connection.send(state, index)

    def gather(self, index: int = 0) -> List[NDArray[np.float64]]:
        return [
            connection.recv(index)
            for connection in self._connections.values()
            if not connection.empty(index)
        ]

    def close(self):
        for connection in self._connections.values():
            while not connection.empty():
                connection.recv()
            connection.close()


def create_async_network(
    node_names: List[str],
    edge_pairs: List[Tuple[str, str]],
    noise_scale: int | float | None = None,
    maxsize: int = 10,
    n_channels: int = 1,
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
        channels_u = [
            Channel(Queue(maxsize=maxsize), Queue(maxsize=maxsize))
            for _ in range(n_channels)
        ]
        channels_v = [
            Channel(channel.out_queue, channel.in_queue) for channel in channels_u
        ]
        conn_u = Connection(*channels_u)
        conn_v = Connection(*channels_v)
        gossip_map[u].add_connection(v, conn_u)
        gossip_map[v].add_connection(u, conn_v)

    return gossip_map
