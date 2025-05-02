import threading as th
import multiprocessing as mp
import numpy as np
from typing import Dict, NamedTuple, List, Tuple
from multiprocessing.synchronize import Event
from multiprocessing.managers import ListProxy
from queue import Queue
from numpy.typing import NDArray
from .gossip import Gossip


class Message(NamedTuple):
    data: ListProxy
    event: Event


class NodeHandle:
    def __init__(self, stop_event: Event = None):
        self.stop_event = stop_event if stop_event else mp.Event()
        self._manager = mp.Manager()
        self._topic_dict: Dict[str, Message] = self._manager.dict()

    def register(self, topic: str):
        if topic not in self._topic_dict:
            self._topic_dict[topic] = Message(
                data=self._manager.list([0.0]),
                event=self._manager.Event(),
            )
        else:
            print(f"Topic '{topic}' already registered.")

    def stop(self):
        self.stop_event.set()
        print("Stopping NodeHandler.")


class Publisher:
    def __init__(self, node_handler: NodeHandle, topic: str, queue_size: int = 10):
        node_handler.register(topic)
        self._topic_dict = node_handler._topic_dict
        self._topic = topic
        self._queue = Queue(maxsize=queue_size)

    def publish(self, data: NDArray[np.float64]):
        if self._queue.full():
            self._queue.get()
        self._queue.put(data)

    def backend(self):
        while True:
            if self._queue.empty():
                continue
            message = self._topic_dict[self._topic]
            message.data[:] = self._queue.get()
            message.event.set()

    def init(self):
        thead = th.Thread(target=self.backend, daemon=True)
        thead.start()


class Subscriber:
    def __init__(
        self,
        node_handler: NodeHandle,
        topic: str,
        queue_size: int = 10,
    ):
        self._topic_dict = node_handler._topic_dict
        self._topic = topic
        self._queue: Queue[List[np.float64]] = Queue(maxsize=queue_size)

    def subscribe(self) -> List[np.float64] | None:
        if self._queue.empty():
            return None
        return self._queue.get()

    def backend(self):
        while True:
            message = self._topic_dict[self._topic]
            message.event.wait()
            if self._queue.full():
                self._queue.get()
            self._queue.put(message.data)
            message.event.clear()

    def init(self):
        thead = th.Thread(target=self.backend, daemon=True)
        thead.start()


class AsyncGossip(Gossip):
    def __init__(
        self,
        node_handle: NodeHandle,
        name: str,
        neighbor_names: List[str] = None,
        noise_scale: int | float | None = None,
        maxsize: int = 10,
        n_channels: int = 1,
    ):
        super().__init__(name, noise_scale)

        self._n_channels = n_channels

        self._publishers = [
            Publisher(node_handle, name, maxsize) for _ in range(n_channels)
        ]
        if neighbor_names is None:
            self._neighbor_names = []
            self._subscribers: List[Dict[str, Subscriber]] = [
                {} for _ in range(n_channels)
            ]
        else:
            self._neighbor_names = neighbor_names
            self._subscribers = [
                {neighbor: Subscriber(node_handle, neighbor, maxsize)}
                for neighbor in neighbor_names
            ]

    @property
    def degree(self) -> int:
        return len(self._subscribers)

    @property
    def neighbor_names(self) -> List[str]:
        return self._neighbor_names

    def add_subscriber(self, neighbor: str, subscriber: Subscriber):
        if neighbor in self._subscribers:
            raise ValueError(f"Subscriber to neighbor '{neighbor}' already exists")
        self._subscribers[neighbor] = subscriber

    def _broadcast_with_noise(self, state, index=0):
        noise = np.random.normal(scale=self._noise_scale, size=state.shape)
        self._publishers[index].publish(state + noise)

    def _broadcast_without_noise(self, state, index=0):
        self._publishers[index].publish(state)

    def gather(self, index: int = 0) -> List[NDArray[np.float64]]:
        states = []
        for subscriber in self._subscribers[index].values():
            received_data = subscriber.subscribe()
            if received_data is not None:
                states.append(np.array(received_data))
        return states

    def init(self):
        for i in range(self._n_channels):
            self._publishers[i].init()
            for subscriber in self._subscribers[i].values():
                subscriber.init()
        print(f"Gossip {self._name} initialized.")


def create_async_network(
    node_handle: NodeHandle,
    node_names: List[str],
    edge_pairs: List[tuple[str, str]],
    noise_scale: int | float | None = None,
    maxsize: int = 10,
    n_channels: int = 1,
) -> Dict[str, Gossip]:
    neighbor_names_dict: Dict[str, List[str]] = {name: [] for name in node_names}

    for u, v in edge_pairs:
        neighbor_names_dict[u].append(v)
        neighbor_names_dict[v].append(u)

    network: Dict[str, Gossip] = {
        name: AsyncGossip(
            node_handle,
            name,
            neighbor_names_dict[name],
            noise_scale,
            maxsize,
            n_channels,
        )
        for name in node_names
    }

    return network
