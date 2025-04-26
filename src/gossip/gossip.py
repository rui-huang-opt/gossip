import numpy as np
from numpy.typing import NDArray
from numpy.random import normal
from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Dict, KeysView, NamedTuple
from multiprocessing import Pipe, Queue
from multiprocessing.connection import Connection


class Gossip(metaclass=ABCMeta):
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

    @property
    @abstractmethod
    def degree(self) -> int: ...

    @property
    @abstractmethod
    def neighbor_names(self) -> KeysView[str]: ...

    @abstractmethod
    def send(self, name: str, state: NDArray[np.float64]): ...

    @abstractmethod
    def recv(self, name: str) -> NDArray[np.float64]: ...

    @abstractmethod
    def _broadcast_with_noise(self, state: NDArray[np.float64]): ...

    @abstractmethod
    def _broadcast_without_noise(self, state: NDArray[np.float64]): ...

    def broadcast(self, state: NDArray[np.float64]):
        if self._noise_scale:
            self._broadcast_with_noise(state)
        else:
            self._broadcast_without_noise(state)

    @abstractmethod
    def gather(self) -> List[NDArray[np.float64]]: ...

    @abstractmethod
    def add_connection(self, neighbor: str, *args, **kwargs): ...

    @abstractmethod
    def remove_connection(self, neighbor: str): ...

    @abstractmethod
    def close(self): ...

    def compute_laplacian(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        self.broadcast(state)
        neighbor_states = self.gather()

        return len(neighbor_states) * state - sum(neighbor_states)


class SyncGossip(Gossip):
    """
    Synchronous gossip communication using multiprocessing pipes.
    This class implements the Gossip interface for synchronous communication.
    """

    def __init__(self, name: str, noise_scale: int | float | None = None):
        super().__init__(name, noise_scale)
        self._connections: Dict[str, Connection] = {}

    @property
    def degree(self) -> int:
        return len(self._connections)

    @property
    def neighbor_names(self) -> KeysView[str]:
        return self._connections.keys()

    def send(self, name: str, state: NDArray[np.float64]):
        if self._noise_scale:
            noise = normal(scale=self._noise_scale, size=state.shape)
            self._connections[name].send(state + noise)
        else:
            self._connections[name].send(state)

    def recv(self, name: str) -> NDArray[np.float64]:
        return self._connections[name].recv()

    def _broadcast_with_noise(self, state: NDArray[np.float64]):
        for conn in self._connections.values():
            noise = normal(scale=self._noise_scale, size=state.shape)
            conn.send(state + noise)

    def _broadcast_without_noise(self, state: NDArray[np.float64]):
        for conn in self._connections.values():
            conn.send(state)

    def gather(self) -> List[NDArray[np.float64]]:
        return [conn.recv() for conn in self._connections.values()]

    def add_connection(self, neighbor: str, conn: Connection):
        self._connections[neighbor] = conn

    def remove_connection(self, neighbor: str):
        self._connections.pop(neighbor, None)

    def close(self):
        for conn in self._connections.values():
            while conn.poll():
                conn.recv()
            conn.close()


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


def create_gossip_network(
    node_names: List[str],
    edge_pairs: List[Tuple[str, str]],
    noise_scale: int | float | None = None,
    is_async: bool = False,
    maxsize: int = 10,
) -> Dict[str, Gossip]:
    """
    Create a gossip network from a list of nodes and edges.

    Parameters
    ----------
    node_names : List[str]
        A list of node identifiers.
    edge_pairs : List[Tuple[str, str]]
        A list of pairs of node identifiers representing edges in the network.
    noise_scale : int | float | None, optional
        The scale of noise to add to messages, by default None.
    is_async : bool, optional
        Whether to use asynchronous gossip, by default False.
    maxsize : int, optional
        The maximum size of the queue for asynchronous gossip, by default 10.

    Returns
    -------
    Dict[str, Gossip]
        A dictionary of gossip communicators indexed by node identifier.
    """

    if is_async:
        gossip_map = {name: AsyncGossip(name, noise_scale) for name in node_names}

        for u, v in edge_pairs:
            queue_u, queue_v = Queue(maxsize=maxsize), Queue(maxsize=maxsize)
            gossip_map[u].add_connection(v, queue_u, queue_v)
            gossip_map[v].add_connection(u, queue_v, queue_u)
    else:
        gossip_map = {name: SyncGossip(name, noise_scale) for name in node_names}

        for u, v in edge_pairs:
            conn_u, conn_v = Pipe()
            gossip_map[u].add_connection(v, conn_u)
            gossip_map[v].add_connection(u, conn_v)

    return gossip_map
