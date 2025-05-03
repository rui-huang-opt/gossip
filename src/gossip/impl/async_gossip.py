import multiprocessing as mp
import numpy as np
from typing import Dict, List, KeysView
from multiprocessing.synchronize import Event
from numpy.typing import NDArray
from ..gossip import Gossip


class NodeHandle:
    def __init__(self, stop_event: Event = None):
        self._stop_event = stop_event if stop_event else mp.Event()
        self._dict: Dict[str, List[mp.Queue]] = {}

    def register_publisher(self, topic: str) -> List[mp.Queue]:
        if topic not in self._dict:
            self._dict[topic] = []
        else:
            print(f"Topic '{topic}' already registered as publisher.")

        return self._dict[topic]

    def register_subscriber(self, topic: str, maxsize: int) -> mp.Queue:
        q = mp.Queue(maxsize=maxsize)
        if topic not in self._dict:
            self._dict[topic] = [q]
        else:
            self._dict[topic].append(q)

        return q

    @property
    def stop_event(self) -> Event:
        return self._stop_event

    def stop(self):
        self._stop_event.set()
        print("Stopping NodeHandler.")


class Publisher:
    def __init__(self, node_handle: NodeHandle, topic: str):
        self._q_list = node_handle.register_publisher(topic)
        self._topic = topic

    def publish(self, data: NDArray[np.float64]):
        for q in self._q_list:
            if q.full():
                q.get()
            q.put(data)


class Subscriber:
    def __init__(
        self,
        node_handle: NodeHandle,
        topic: str,
        maxsize: int = 10,
    ):
        self._q = node_handle.register_subscriber(topic, maxsize)
        self._topic = topic

    def update(self) -> NDArray[np.float64] | None:
        if self._q.empty():
            return None
        return self._q.get()


class AsyncGossip(Gossip):
    def __init__(
        self,
        node_handle: NodeHandle,
        name: str,
        neighbor_names: List[str],
        noise_scale: int | float | None = None,
        maxsize: int = 10,
        n_channels: int = 1,
    ):
        super().__init__(name, noise_scale)

        self._n_channels = n_channels

        self._publishers = [
            Publisher(node_handle, f"{name}/{i}") for i in range(n_channels)
        ]
        self._subscribers = [
            {
                neighbor: Subscriber(node_handle, f"{neighbor}/{i}", maxsize)
                for neighbor in neighbor_names
            }
            for i in range(n_channels)
        ]

    @property
    def degree(self) -> int:
        return len(self._subscribers[0])

    @property
    def neighbor_names(self) -> KeysView[str]:
        return self._subscribers[0].keys()

    def _broadcast_with_noise(self, state, index=0):
        noise = np.random.normal(scale=self._noise_scale, size=state.shape)
        self._publishers[index].publish(state + noise)

    def _broadcast_without_noise(self, state, index=0):
        self._publishers[index].publish(state)

    def gather(self, index: int = 0) -> List[NDArray[np.float64]]:
        data = []
        for subscriber in self._subscribers[index].values():
            received_data = subscriber.update()
            if received_data is not None:
                data.append(received_data)
        return data

    def compute_laplacian(
        self, state: NDArray[np.float64], index: int = 0
    ) -> NDArray[np.float64]:
        self.broadcast(state, index)
        neighbor_states = self.gather(index)
        return len(neighbor_states) * state - sum(neighbor_states)


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
