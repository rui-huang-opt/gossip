import numpy as np
from numpy.typing import NDArray
from numpy.random import normal
from typing import KeysView, Mapping, overload
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from ..gossip import Gossip


class SyncGossip(Gossip):
    """
    Synchronous gossip communication using multiprocessing pipes.
    This class implements the Gossip interface for synchronous communication.
    """

    def __init__(
        self,
        name: int | str,
        noise_scale: float | None = None,
    ):
        super().__init__(name, noise_scale)
        self._connections: dict[str, Connection] | dict[int, Connection] = {}

    @property
    def degree(self) -> int:
        return len(self._connections)

    @property
    def neighbor_names(self) -> KeysView[int] | KeysView[str]:
        return self._connections.keys()

    def send(self, name, state, index=0):
        if name not in self._connections:
            raise ValueError(f"Connection to {name} does not exist.")
        if self._noise_scale is not None:
            noise = normal(scale=self._noise_scale, size=state.shape)
            state = state + noise
        else:
            self._connections[name].send(state)

    def recv(self, name, index=0) -> NDArray[np.float64]:
        if name not in self._connections:
            raise ValueError(f"Connection to {name} does not exist.")
        return self._connections[name].recv()

    def broadcast(self, state: NDArray[np.float64], index: int = 0):
        if self._noise_scale is None:
            for conn in self._connections.values():
                conn.send(state)
        else:
            for conn in self._connections.values():
                noise = normal(scale=self._noise_scale, size=state.shape)
                conn.send(state + noise)

    def gather(self, index: int = 0) -> list[NDArray[np.float64]]:
        return [conn.recv() for conn in self._connections.values()]

    @overload
    def add_connection(self, name: str, conn: Connection): ...

    @overload
    def add_connection(self, name: int, conn: Connection): ...

    def add_connection(self, name, conn):
        if name in self._connections:
            return
        self._connections[name] = conn


@overload
def create_sync_network(
    node_names: list[str],
    edge_pairs: list[tuple[str, str]],
    noise_scale: float | None = None,
) -> dict[str, SyncGossip]: ...


@overload
def create_sync_network(
    node_names: list[int],
    edge_pairs: list[tuple[int, int]],
    noise_scale: float | None = None,
) -> dict[int, SyncGossip]: ...


def create_sync_network(
    node_names,
    edge_pairs,
    noise_scale=None,
):
    network = {name: SyncGossip(name, noise_scale) for name in node_names}

    for u, v in edge_pairs:
        conn_u, conn_v = Pipe()
        network[u].add_connection(v, conn_u)
        network[v].add_connection(u, conn_v)

    return network
