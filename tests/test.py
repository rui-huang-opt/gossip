import time
import zmq
import multiprocessing as mp
import numpy as np
from numpy.typing import NDArray


class Server(mp.Process):
    def __init__(self):
        super().__init__()

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.bind("ipc://@test")

        for request in range(10):
            socket.send(b"Hello World")
            time.sleep(1)


class Client(mp.Process):
    def __init__(self):
        super().__init__()

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.connect("ipc://@test")

        for request in range(10):
            message = socket.recv()
            print(f"Received reply {request} [ {message.decode()} ]")
            time.sleep(1)


class Gossip:
    def __init__(self, name: str, dim: int = 3):
        self._name = name
        self._dim = dim
        self._dtype = np.float64

        self._context = zmq.Context()

        self._pull_socket = self._context.socket(zmq.PULL)
        self._pull_socket.bind(f"ipc://@{name}")
        self._push_sockets: dict[str, zmq.SyncSocket] = {}

    @property
    def degree(self) -> int:
        return len(self._push_sockets)

    def add_neighbor(self, neighbor: str):
        if neighbor in self._push_sockets:
            return
        self._push_sockets[neighbor] = self._context.socket(zmq.PUSH)
        self._push_sockets[neighbor].connect(f"ipc://@{neighbor}")

    def close(self):
        self._pull_socket.close(linger=0)
        for socket in self._push_sockets.values():
            socket.close(linger=0)
        self._context.term()

    def broadcast(self, data: NDArray[np.float64]):
        for socket in self._push_sockets.values():
            socket.send(data.tobytes())

    def gather(self) -> list[NDArray[np.float64]]:
        msg_list = []
        for _ in range(self.degree):
            data = self._pull_socket.recv()
            arr = np.frombuffer(data, dtype=self._dtype).reshape((self._dim,))
            msg_list.append(arr)
        return msg_list

    def compute_laplacian(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        self.broadcast(state)
        neighbor_states = self.gather()

        return state * len(neighbor_states) - sum(neighbor_states)


class Node(mp.Process):
    def __init__(
        self,
        name: str,
        neighbor_names: list[str],
        state: NDArray[np.float64] | None = None,
    ):
        super().__init__()
        self._name = name
        self._neighbor_names = neighbor_names
        self._state = np.zeros((3,), dtype=np.float64) if state is None else state

    def run(self):
        conn = Gossip(self._name)
        for neighbor in self._neighbor_names:
            conn.add_neighbor(neighbor)

        for k in range(500):
            error = conn.compute_laplacian(self._state)
            self._state -= error * 0.45
            print(f"Node {self._name} state: {self._state}")

        conn.close()

        print(f"Node {self._name} finished with steps: {k}")


if __name__ == "__main__":
    states = {
        "node_0": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "node_1": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "node_2": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "node_3": np.array([1.0, 1.0, 1.0], dtype=np.float64),
        "node_4": np.array([2.0, 2.0, 2.0], dtype=np.float64),
        "node_5": np.array([3.0, 3.0, 3.0], dtype=np.float64),
        "node_6": np.array([4.0, 4.0, 4.0], dtype=np.float64),
        "node_7": np.array([5.0, 5.0, 5.0], dtype=np.float64),
        "node_8": np.array([6.0, 6.0, 6.0], dtype=np.float64),
        "node_9": np.array([7.0, 7.0, 7.0], dtype=np.float64),
    }

    node_names = list(states.keys())
    edge_pairs = [
        ("node_0", "node_1"),
        ("node_1", "node_2"),
        ("node_2", "node_3"),
        ("node_3", "node_4"),
        ("node_4", "node_5"),
        ("node_5", "node_6"),
        ("node_6", "node_7"),
        ("node_7", "node_8"),
        ("node_8", "node_9"),
        ("node_9", "node_0"),
        ("node_0", "node_2"),
    ]

    topology = {i: [] for i in node_names}
    for u, v in edge_pairs:
        topology[u].append(v)
        topology[v].append(u)

    nodes = {i: Node(i, topology[i], states[i]) for i in node_names}

    for node in nodes.values():
        node.start()

    for node in nodes.values():
        node.join()
