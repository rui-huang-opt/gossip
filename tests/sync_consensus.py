import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict
from numpy.typing import NDArray
from multiprocessing import Process
from gossip import Gossip, create_sync_network


class Node(Process):
    def __init__(
        self,
        comm: Gossip,
        initial_state: NDArray[np.float64],
        max_iter: int,
        step_size: float,
        results_path: str,
    ):
        super().__init__()

        self.comm = comm
        self.state = np.tile(initial_state, (max_iter + 1, 1))

        self.max_iter = max_iter
        self.step_size = step_size
        self.results_path = results_path

    def run(self):
        for k in range(self.max_iter):
            delta_state = self.comm.compute_laplacian(self.state[k])

            self.state[k + 1] = self.state[k] - self.step_size * delta_state

        os.makedirs(self.results_path, exist_ok=True)
        np.save(
            os.path.join(self.results_path, f"node_{self.comm.name}.npy"), self.state
        )


if __name__ == "__main__":
    script_type = "plot"  # "test" or "plot"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results", "sync_consensus")

    node_names = ["1", "2", "3", "4", "5"]
    edge_pairs = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "1")]

    if script_type == "test":
        initial_states = {
            "1": np.array([10.1, 20.2, 30.3]),
            "2": np.array([52.3, 42.2, 32.1]),
            "3": np.array([25.6, 35.5, 45.4]),
            "4": np.array([17.7, 27.6, 37.5]),
            "5": np.array([20.9, 30.8, 40.7]),
        }
        node_params = {"max_iter": 50, "step_size": 0.5, "results_path": results_dir}
        gossip_network = create_sync_network(node_names, edge_pairs, noise_scale=0.1)

        consensus_nodes = [
            Node(gossip_network[name], initial_states[name], **node_params)
            for name in node_names
        ]

        for node in consensus_nodes:
            node.start()

        for node in consensus_nodes:
            node.join()

    elif script_type == "plot":
        figure_dir = os.path.join(script_dir, "figures", "sync_consensus")
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

        states_dict: Dict[str, NDArray[np.float64]] = {
            name: np.load(os.path.join(results_dir, f"node_{name}.npy"))
            for name in node_names
        }

        for name in node_names:
            for i in range(3):
                (line,) = ax2.plot(
                    states_dict[name][:, i],
                    color=node_colors[name],
                )

            line.set_label(f"Node {name}")

        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("State")

        ax2.legend(loc="upper right")

        ax2.set_title("Synchronous Consensus")

        fig2.savefig(
            os.path.join(figure_dir, "sync_consensus.png"), dpi=300, bbox_inches="tight"
        )
