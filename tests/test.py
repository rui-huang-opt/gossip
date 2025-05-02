import time
import threading as th
import multiprocessing as mp
import numpy as np
from typing import Dict, NamedTuple, List
from multiprocessing.synchronize import Event
from multiprocessing.managers import ListProxy
from queue import Queue
from numpy.typing import NDArray


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

    def clear(self):
        while not self._queue.empty():
            self._queue.get()

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

    def clear(self):
        while not self._queue.empty():
            self._queue.get()

    def init(self):
        thead = th.Thread(target=self.backend, daemon=True)
        thead.start()


class Gossip:
    def __init__(
        self,
        node_handle: NodeHandle,
        name: str,
        neighbor_names: List[str] = None,
        noise_scale: int | float | None = None,
        maxsize: int = 10,
        n_channels: int = 1,
    ):
        self._name = name
        self._noise_scale = noise_scale
        self._n_channels = n_channels

        self._publishers = [
            Publisher(node_handle, name, maxsize) for _ in range(n_channels)
        ]
        if neighbor_names is None:
            self._subscribers: List[Dict[str, Subscriber]] = [
                {} for _ in range(n_channels)
            ]
        else:
            self._subscribers = [
                {neighbor: Subscriber(node_handle, neighbor, maxsize)}
                for neighbor in neighbor_names
            ]

    def add_subscriber(self, neighbor: str, subscriber: Subscriber):
        if neighbor in self._subscribers:
            raise ValueError(f"Subscriber to neighbor '{neighbor}' already exists")
        self._subscribers[neighbor] = subscriber

    def broadcast(self, data: NDArray[np.float64], index: int = 0):
        if self._noise_scale:
            noise = np.random.normal(scale=self._noise_scale, size=data.shape)
            self._publishers[index].publish(data + noise)
        else:
            self._publishers[index].publish(data)

    def gather(self, index: int = 0) -> List[NDArray[np.float64]]:
        data = []
        for subscriber in self._subscribers[index].values():
            received_data = subscriber.subscribe()
            if received_data is not None:
                data.append(np.array(received_data))
        return data

    def compute_laplacian(
        self, state: NDArray[np.float64], index: int = 0
    ) -> NDArray[np.float64]:
        self.broadcast(state, index)
        neighbor_states = self.gather(index)
        return len(neighbor_states) * state - sum(neighbor_states)

    def clear(self):
        for i in range(self._n_channels):
            self._publishers[i].clear()
            for subscriber in self._subscribers[i].values():
                subscriber.clear()

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
        name: Gossip(
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


class ConsensusNode(mp.Process):
    def __init__(
        self,
        name: str,
        communicator: Gossip,
        initial_state: NDArray[np.float64],
        global_stop_event: Event,
        step_size: float,
        private_stop_event: Event = None,
    ):
        super().__init__()

        self._name = name
        self._communicator = communicator
        self._state = initial_state
        self._global_stop_event = global_stop_event
        self._private_stop_event = (
            private_stop_event if private_stop_event else mp.Event()
        )
        self._step_size = step_size
        self._state_list: List[NDArray[np.float64]] = []

    def init(self):
        self._communicator.init()
        print(f"Node {self._name} initialized.")

    def main(self):
        while not self._global_stop_event.is_set():
            delta_state = self._communicator.compute_laplacian(self._state)
            self._state -= self._step_size * delta_state

            print(f"Node {self._name} state: {self._state}")

            time.sleep(0.001)

            if self._private_stop_event.is_set():
                break

        print(f"Node {self._name} finished.")

    def run(self):
        self.init()
        self.main()

    def shutdown(self):
        self._private_stop_event.set()
        self.join()


if __name__ == "__main__":
    nh = NodeHandle()

    node_names_ = ["1", "2", "3", "4", "5"]
    edge_pairs_ = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "1")]
    initial_states = {
        "1": np.array([10.1, 20.2, 30.3]),
        "2": np.array([52.3, 42.2, 32.1]),
        "3": np.array([25.6, 35.5, 45.4]),
        "4": np.array([17.7, 27.6, 37.5]),
        "5": np.array([20.9, 30.8, 40.7]),
    }
    gossip_network = create_async_network(nh, node_names_, edge_pairs_)
    alpha = 0.5

    nodes = {
        name: ConsensusNode(
            name,
            gossip_network[name],
            initial_states[name],
            nh.stop_event,
            alpha,
        )
        for name in node_names_
    }

    for node in nodes.values():
        node.start()

    time.sleep(0.2)
    nodes["1"].shutdown()
    time.sleep(0.2)
    nodes["1"] = ConsensusNode(
        "1",
        gossip_network["1"],
        initial_states["1"],
        nh.stop_event,
        alpha,
    )
    nodes["1"].start()
    time.sleep(1)
    nh.stop()

    for node in nodes.values():
        node.join()

    print("All nodes have finished.")
