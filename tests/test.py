import time
import zmq
from multiprocessing import Process
from threading import Thread, Event
from functools import cached_property
from numpy import float64, array, frombuffer
from numpy.typing import NDArray
from numpy.random import seed, uniform


class SyncGossip:
    def __init__(
        self,
        name: str,
        neighbors: list[str] | None = None,
        noise_scale: float | None = None,
    ):
        self._name = name
        self._neighbors = neighbors if neighbors is not None else []
        self._noise_scale = noise_scale
        self._context = zmq.Context()

        self._router = self._context.socket(zmq.ROUTER)
        self._router.bind(f"ipc://@{self._name}")

        self._sockets: dict[str, zmq.SyncSocket] = {}
        for neighbor in self._neighbors:
            socket = self._context.socket(zmq.REQ)
            socket.connect(f"ipc://@{neighbor}")
            self._sockets[neighbor] = socket

    @property
    def degree(self) -> int:
        return len(self._neighbors)

    def warmup(self):
        clients: list[bytes] = []

        for socket in self._sockets.values():
            socket.send(b"")

        for _ in range(self.degree):
            client, _, _ = self._router.recv_multipart()
            clients.append(client)

        for client in clients:
            self._router.send_multipart([client, b"", b""])

        for socket in self._sockets.values():
            socket.recv()

    def exchange(self, state: NDArray[float64]) -> list[NDArray[float64]]:
        request = state.tobytes()
        for socket in self._sockets.values():
            socket.send(request)

        requests = [self._router.recv_multipart() for _ in range(self.degree)]
        neighbor_states = [
            frombuffer(n_state, dtype=float64) for _, _, n_state in requests
        ]
        clients = [client for client, _, _ in requests]

        reply = b"received"
        for client in clients:
            self._router.send_multipart([client, b"", reply])

        for socket in self._sockets.values():
            _ = socket.recv()

        return neighbor_states


class Peer(Process):
    def __init__(self, name: str, neighbors: list[str], state: NDArray[float64]):
        super().__init__()

        self._name = name
        self._neighbors = neighbors
        self._context = zmq.Context()
        self._state = state.astype(float64)
        self._max_iter = 10000

    @cached_property
    def degree(self) -> int:
        return len(self._neighbors)

    def run_req_router(self):
        comm = SyncGossip(self._name, self._neighbors)
        comm.warmup()  # Not necessary for the first iteration, here for testing the performance

        begin = time.perf_counter()
        for k in range(self._max_iter):
            # print(f"{self._name} iteration {k}: {self._state}")

            neighbor_states = comm.exchange(self._state)

            bias = self._state * self.degree - sum(neighbor_states)
            self._state -= bias * 0.45

        end = time.perf_counter()
        print(f"{self._name} finished in {end - begin:.6f} seconds")

        print(f"{self._name} final state: {self._state}")

    def run_push_pull(self):
        pusher = self._context.socket(zmq.PUSH)
        pusher.bind(f"ipc://@{self._name}")

        pullers: dict[str, zmq.SyncSocket] = {}
        for neighbor in self._neighbors:
            puller = self._context.socket(zmq.PULL)
            puller.connect(f"ipc://@{neighbor}")
            pullers[neighbor] = puller

        time.sleep(1)  # Allow time for the sockets to connect

        begin = time.perf_counter()

        for k in range(self._max_iter):
            # print(f"{self._name} iteration {k}: {self._state}")

            state_bytes = self._state.tobytes()
            for _ in range(self.degree):
                pusher.send_multipart([str(k).encode(), state_bytes])

            msgs = [puller.recv_multipart() for puller in pullers.values()]
            prefix_list = [prefix.decode() for prefix, _ in msgs]
            neighbor_states = [frombuffer(state, dtype=float64) for _, state in msgs]

            bias = self._state * self.degree - sum(neighbor_states)
            self._state -= bias * 0.45

            sync = all(prefix == str(k) for prefix in prefix_list)
            if not sync:
                print(f"{self._name} not synchronized at iteration {k}")

        end = time.perf_counter()
        print(f"{self._name} finished in {end - begin:.6f} seconds")

        print(f"{self._name} final state: {self._state}")

    def run(self):
        # self.run_req_router()
        self.run_push_pull()


if __name__ == "__main__":
    node_names = ["1", "2", "3", "4", "5"]
    edge_pairs = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "1")]

    seed(0)
    states = {name: uniform(-10.0, 10.0, size=3) for name in node_names}

    network = {name: [] for name in node_names}
    for a, b in edge_pairs:
        network[a].append(b)
        network[b].append(a)

    nodes = [Peer(name, neighbors, states[name]) for name, neighbors in network.items()]

    for node in nodes:
        node.start()

    for node in nodes:
        node.join()
