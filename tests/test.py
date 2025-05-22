import time
import zmq
from multiprocessing import Process
from functools import cached_property


class Peer(Process):
    def __init__(self, name: str, neighbors: list[str]):
        super().__init__()

        self._name = name
        self._neighbors = neighbors
        self._context = zmq.Context()

    @cached_property
    def degree(self) -> int:
        return len(self._neighbors)

    def run(self):
        router = self._context.socket(zmq.ROUTER)
        router.bind(f"ipc://@{self._name}")

        sockets: dict[str, zmq.SyncSocket] = {}
        for neighbor in self._neighbors:
            socket = self._context.socket(zmq.REQ)
            socket.connect(f"ipc://@{neighbor}")
            sockets[neighbor] = socket

        for k in range(10):
            msg_list: list[str] = []
            for neighbor, socket in sockets.items():
                request = b"iteration " + str(k).encode()
                socket.send(request)

            for _ in range(self.degree):
                client, _, request = router.recv_multipart()
                reply = f"My name is {self._name}, I received: {request.decode()}"
                router.send_multipart([client, b"", reply.encode()])

            for neighbor, socket in sockets.items():
                reply = socket.recv()
                msg_list.append(reply.decode())

            sync = all([msg.endswith(f"iteration {k}") for msg in msg_list])
            print(f"Sync status for {self._name} at iteration {k}: {sync}")


if __name__ == "__main__":
    node_names = [f"node_{i}" for i in range(10)]
    edge_pairs = [
        ("node_0", "node_1"),
        ("node_0", "node_2"),
        ("node_1", "node_3"),
        ("node_1", "node_4"),
        ("node_2", "node_5"),
        ("node_2", "node_6"),
        ("node_3", "node_7"),
        ("node_4", "node_8"),
        ("node_5", "node_9"),
    ]

    network = {name: [] for name in node_names}
    for a, b in edge_pairs:
        network[a].append(b)
        network[b].append(a)

    nodes = [Peer(name, neighbors) for name, neighbors in network.items()]

    for node in nodes:
        node.start()

    for node in nodes:
        node.join()
