import time
import threading as th
import multiprocessing as mp
import numpy as np
from typing import Dict, NamedTuple, Callable, List
from multiprocessing.synchronize import Event
from multiprocessing.managers import ListProxy
from queue import Queue
from numpy.typing import NDArray


class Message(NamedTuple):
    dim: int
    data: ListProxy
    event: Event


class NodeHandle:
    def __init__(self, stop_event: Event = None):
        self.stop_event = stop_event if stop_event else mp.Event()
        self._manager = mp.Manager()
        self._topic_dict: Dict[str, Message] = self._manager.dict()

    def register_topic(self, topic: str, dim: int):
        if topic not in self._topic_dict:
            self._topic_dict[topic] = Message(
                dim=dim,
                data=self._manager.list([0.0] * dim),
                event=self._manager.Event(),
            )
        else:
            print(f"Topic '{topic}' already registered.")

    def stop(self):
        self.stop_event.set()
        print("Stopping NodeHandler.")


class Publisher:
    def __init__(
        self, node_handler: NodeHandle, topic: str, dim: int = 3, queue_size: int = 10
    ):
        node_handler.register_topic(topic, dim)
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
        callback: Callable[[Queue[List[np.float64]]], None],
        queue_size: int = 10,
    ):
        self._topic_dict = node_handler._topic_dict
        self._topic = topic
        self._callback = callback
        self._queue = Queue(maxsize=queue_size)

    def backend(self):
        while True:
            message = self._topic_dict[self._topic]
            message.event.wait()
            if self._queue.full():
                self._queue.get()
            self._queue.put(message.data)
            self._callback(self._queue)
            message.event.clear()

    def init(self):
        thead = th.Thread(target=self.backend, daemon=True)
        thead.start()


class PubNode(mp.Process):
    def __init__(self, node_handler: NodeHandle, dim: int = 3):
        super().__init__()

        self._dim = dim
        self._publisher = Publisher(node_handler, "array", dim=dim)
        self._stop_event = node_handler.stop_event

    def init(self):
        self._publisher.init()
        print("Publisher initialized.")

    def main(self):
        while not self._stop_event.is_set():
            data = np.random.rand(self._dim)
            self._publisher.publish(data)
            time.sleep(0.01)
        print("Publisher stopped.")

    def run(self):
        self.init()
        self.main()


class SubNode(mp.Process):
    def __init__(self, node_handler: NodeHandle):
        super().__init__()

        self._subscriber = Subscriber(node_handler, "array", lambda q: print(q.get()))

        self._stop_event = node_handler.stop_event

    def init(self):
        self._subscriber.init()
        print("Subscriber initialized.")

    def main(self):
        while not self._stop_event.is_set():
            time.sleep(0.1)
        print("Subscriber stopped.")

    def run(self):
        self.init()
        self.main()


if __name__ == "__main__":
    nh = NodeHandle()
    pub_node = PubNode(nh, 50)
    sub_node = SubNode(nh)
    pub_node.start()
    sub_node.start()

    time.sleep(5)
    nh.stop()

    pub_node.join()
    sub_node.join()

    print("Main process finished.")
