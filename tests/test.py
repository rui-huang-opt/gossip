import time
import threading as th
import multiprocessing as mp
import numpy as np
from typing import Dict, NamedTuple, Callable
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event
from multiprocessing.managers import ListProxy
from queue import Queue
from numpy.typing import NDArray


class Message(NamedTuple):
    data: ListProxy
    event: Event


class NodeHandler:
    def __init__(self, stop_event: Event):
        self.stop_event = stop_event
        self._manager = mp.Manager()
        self._topic_dict: Dict[str, Message] = self._manager.dict()

    def register_topic(self, topic: str, dim: int):
        if topic not in self._topic_dict:
            self._topic_dict[topic] = Message(
                data=self._manager.list([0.0] * dim),
                event=self._manager.Event(),
            )
        else:
            print(f"Topic '{topic}' already registered.")

    def stop(self):
        self.stop_event.set()
        print("Stopping NodeHandler.")


class Publisher:
    def __init__(self, node_handler: NodeHandler, topic: str, queue_size: int = 10):
        node_handler.register_topic(topic, 3)
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
        node_handler: NodeHandler,
        topic: str,
        callback: Callable[[Queue[NDArray[np.float64]]], None],
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
            self._queue.put(message.data[:])
            self._callback(self._queue)
            message.event.clear()

    def init(self):
        thead = th.Thread(target=self.backend, daemon=True)
        thead.start()


class PubNode(mp.Process):
    def __init__(self, node_handler: NodeHandler):
        super().__init__()

        self._publisher = Publisher(node_handler, "array")
        self._stop_event = node_handler.stop_event

    def run(self):
        self._publisher.init()
        print("Publisher initialized.")

        while not self._stop_event.is_set():
            data = np.random.rand(3)
            self._publisher.publish(data)
            time.sleep(0.1)
        print("Publisher stopped.")


class SubNode(mp.Process):
    def __init__(self, node_handler: NodeHandler):
        super().__init__()

        self._subscriber = Subscriber(
            node_handler, "array", lambda q: print(q.get())
        )

        self._stop_event = node_handler.stop_event

    def run(self):
        self._subscriber.init()
        print("Subscriber initialized.")

        while not self._stop_event.is_set():
            time.sleep(0.1)
        print("Subscriber stopped.")


if __name__ == "__main__":
    nh = NodeHandler(mp.Event())
    pub_node = PubNode(nh)
    sub_node = SubNode(nh)
    pub_node.start()
    sub_node.start()

    time.sleep(5)
    nh.stop()

    pub_node.join()
    sub_node.join()

    print("Main process finished.")
