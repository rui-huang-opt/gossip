import numpy as np
from numpy.typing import NDArray
from numpy.random import normal
from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Dict, KeysView, TypeVar, Generic
from multiprocessing import Pipe, Queue
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

    @abstractmethod
    def send(self, name: str, state: NDArray[np.float64]): ...

    @abstractmethod
    def recv(self, name: str) -> NDArray[np.float64]: ...

    @abstractmethod
    def broadcast(self, state: NDArray[np.float64]): ...

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

    def broadcast(self, state: NDArray[np.float64]):
        for conn in self._connections.values():
            if self._noise_scale:
                noise = normal(scale=self._noise_scale, size=state.shape)
                conn.send(state + noise)
            else:
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


class AsyncGossip(Gossip):
    def __init__(self, name: str, noise_scale: int | float | None = None):
        super().__init__(name, noise_scale)
        self._in_queues: Dict[str, Queue] = {}
        self._out_queues: Dict[str, Queue] = {}

    def send(self, name: str, state: NDArray[np.float64]):
        queue = self._out_queues.get(name)
        if queue is None:
            raise ValueError(f"No connection to neighbor '{name}'")
        if queue.full():
            queue.get()
        if self._noise_scale:
            noise = normal(scale=self._noise_scale, size=state.shape)
            queue.put(state + noise)
        else:
            queue.put(state)

    def recv(self, name: str) -> NDArray[np.float64] | None:
        queue = self._in_queues.get(name)
        if queue is None:
            raise ValueError(f"No connection from neighbor '{name}'")
        return queue.get() if not queue.empty() else None

    def broadcast(self, state: NDArray[np.float64]):
        for queue in self._out_queues.values():
            if queue.full():
                queue.get()
            if self._noise_scale:
                noise = normal(scale=self._noise_scale, size=state.shape)
                queue.put(state + noise)
            else:
                queue.put(state)

    def gather(self) -> List[NDArray[np.float64]]:
        return [queue.get() for queue in self._in_queues.values() if not queue.empty()]

    def add_connection(self, neighbor: str, in_queue: Queue, out_queue: Queue):
        if neighbor in self._in_queues or neighbor in self._out_queues:
            raise ValueError(f"Connection to neighbor '{neighbor}' already exists")
        self._in_queues[neighbor] = in_queue
        self._out_queues[neighbor] = out_queue

    def remove_connection(self, neighbor: str):
        if neighbor not in self._in_queues or neighbor not in self._out_queues:
            raise ValueError(f"No connection to neighbor '{neighbor}' to remove")
        self._in_queues.pop(neighbor)
        self._out_queues.pop(neighbor)

    def close(self):
        for queue in self._in_queues.values():
            while not queue.empty():
                queue.get()
            queue.close()
        for queue in self._out_queues.values():
            while not queue.empty():
                queue.get()
            queue.close()


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
