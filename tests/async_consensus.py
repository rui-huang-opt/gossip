import os
import time
import multiprocessing as mp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List
from multiprocessing.synchronize import Event, Barrier
from numpy.typing import NDArray
from gossip import Gossip, create_async_network, NodeHandle


class Node(mp.Process):
    def __init__(
        self,
        comm: Gossip,
        initial_state: NDArray[np.float64],
        step_size: float,
        stop_event: Event,
        global_results: Dict[str, List[NDArray[np.float64]]],
        start_barrier: Barrier | None = None,
    ):
        super().__init__()

        self.comm = comm
        self.state = initial_state
        self.time_list: List[float] = []
        self.state_list: List[NDArray[np.float64]] = []

        self.step_size = step_size
        self.stop_event = stop_event
        self.global_results = global_results
        self.start_barrier = start_barrier
        self.local_stop_event = mp.Event()

    def stop(self):
        self.local_stop_event.set()
        self.join()

    def run(self):
        if self.start_barrier is not None:
            self.start_barrier.wait()

        while not self.stop_event.is_set():
            self.time_list.append(time.time())
            self.state_list.append(self.state.copy())
            delta_state = self.comm.compute_laplacian(self.state)
            self.state = self.state - self.step_size * delta_state

            if self.local_stop_event.is_set():
                break
        
        self.time_list.append(np.nan)
        self.state_list.append(np.nan * np.ones_like(self.state))
        self.global_results["time"].extend(self.time_list)
        self.global_results["state"].extend(self.state_list)


if __name__ == "__main__":
    script_type = "plot"  # "test" or "plot"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results", "async_consensus")

    node_names = ["1", "2", "3", "4", "5"]
    edge_pairs = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "1")]

    if script_type == "test":
        m = mp.Manager()
        g_results = m.dict()

        for name in node_names:
            g_results[name] = m.dict()
            g_results[name]["time"] = m.list([])
            g_results[name]["state"] = m.list([])

        barrier = mp.Barrier(len(node_names) + 1)
        nh = NodeHandle()

        initial_states = {
            "1": np.array([10.1, 20.2, 30.3]),
            "2": np.array([52.3, 42.2, 32.1]),
            "3": np.array([25.6, 35.5, 45.4]),
            "4": np.array([17.7, 27.6, 37.5]),
            "5": np.array([20.9, 30.8, 40.7]),
        }
        node_params = {
            "step_size": 0.5,
            "stop_event": nh.stop_event,
        }
        gossip_network = create_async_network(nh, node_names, edge_pairs, maxsize=50)

        consensus_nodes = {
            name: Node(
                gossip_network[name],
                initial_states[name],
                global_results=g_results[name],
                start_barrier=barrier,
                **node_params,
            )
            for name in node_names
        }

        for node in consensus_nodes.values():
            node.start()

        barrier.wait()
        time.sleep(0.1)
        consensus_nodes["2"].stop()
        time.sleep(0.1)
        consensus_nodes[0] = Node(
            gossip_network["2"],
            initial_states["2"],
            global_results=g_results["2"],
            **node_params,
        )
        consensus_nodes[0].start()
        time.sleep(0.1)
        nh.stop()

        for node in consensus_nodes.values():
            node.join()

        os.makedirs(results_dir, exist_ok=True)
        for name in node_names:
            time_arr = np.array(g_results[name]["time"])
            time_arr -= time_arr[0]
            state_arr = np.array(g_results[name]["state"])
            np.savez(
                os.path.join(results_dir, f"node_{name}.npz"),
                time=time_arr,
                state=state_arr,
            )

    elif script_type == "plot":
        figure_dir = os.path.join(script_dir, "figures", "async_consensus")
        os.makedirs(figure_dir, exist_ok=True)

        fig1, ax1 = plt.subplots()

        G = nx.Graph()
        G.add_nodes_from(node_names)
        G.add_edges_from(edge_pairs)

        node_pos = {
            "1": (0, 0),
            "2": (1, 0),
            "3": (1, 1),
            "4": (0, 1),
            "5": (0.5, 0.5),
        }

        nx.draw(
            G,
            node_pos,
            ax=ax1,
            with_labels=True,
            node_size=1000,
            node_color="skyblue",
            font_size=10,
            font_color="black",
            font_weight="bold",
            edge_color="black",
            width=2,
            style="dashed",
        )

        ax1.set_title("Graph G")

        fig1.savefig(
            os.path.join(figure_dir, "graph.png"), dpi=300, bbox_inches="tight"
        )

        node_colors = {
            "1": "red",
            "2": "blue",
            "3": "green",
            "4": "orange",
            "5": "purple",
        }

        fig2, ax2 = plt.subplots()

        time_dict: Dict[str, NDArray[np.float64]] = {
            name: np.load(os.path.join(results_dir, f"node_{name}.npz"))["time"]
            for name in node_names
        }
        states_dict: Dict[str, NDArray[np.float64]] = {
            name: np.load(os.path.join(results_dir, f"node_{name}.npz"))["state"]
            for name in node_names
        }

        for name in node_names:
            for i in range(3):
                (line,) = ax2.plot(
                    time_dict[name],
                    states_dict[name][:, i],
                    color=node_colors[name],
                )

            line.set_label(f"Node {name}")

        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("State")

        failure_time_indices = np.where(np.isnan(time_dict["2"]))[0]
        ax2.axvspan(
            time_dict["2"][failure_time_indices[0] - 1],
            time_dict["2"][failure_time_indices[0] + 1],
            color="gray",
            alpha=0.3,
            label="Node 2 Failure",
        )

        ax2.legend(loc="upper right", fontsize=8)

        ax2.set_title("Asynchronous Consensus")

        fig2.savefig(
            os.path.join(figure_dir, "async_consensus.png"),
            dpi=300,
            bbox_inches="tight",
        )
