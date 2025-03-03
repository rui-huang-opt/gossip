import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import toml
from typing import Dict
from numpy.typing import NDArray
from multiprocessing import Process
from gossip import create_gossip_network, Gossip


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
            self.comm.broadcast(self.state[k])
            neighbors_state = self.comm.gather()

            consensus_error = self.comm.degree * self.state[k] - sum(neighbors_state)

            self.state[k + 1] = self.state[k] - self.step_size * consensus_error

        os.makedirs(self.results_path, exist_ok=True)
        np.save(self.results_path + f"/node_{self.comm.name}.npy", self.state)


if __name__ == "__main__":
    configs = toml.load("configs/consensus.toml")

    node_names = configs["node_names"]
    edge_pairs = configs["edge_pairs"]

    if configs["run_type"] == "test":
        initial_states  = configs["initial_states"]
        gossip_network = create_gossip_network(node_names, edge_pairs)

        consensus_nodes = [
            Node(
                gossip_network[name],
                initial_states[name],
                **configs["node_params"],
            )
            for name in node_names
        ]

        for node in consensus_nodes:
            node.start()

        for node in consensus_nodes:
            node.join()

    elif configs["run_type"] == "plot":
        figure_path = "figures/consensus/"

        fig1, ax1 = plt.subplots()

        G = nx.Graph()
        G.add_nodes_from(node_names)
        G.add_edges_from(edge_pairs)

        nx.draw(
            G,
            configs["node_pos"],
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

        fig1.savefig(figure_path + "graph.png", dpi=300, bbox_inches="tight")

        node_colors = {
            "1": "red",
            "2": "blue",
            "3": "green",
            "4": "orange",
            "5": "purple",
        }

        fig2, ax2 = plt.subplots()

        states_dict: Dict[str, NDArray[np.float64]] = {
            name: np.load(configs["node_params"]["results_path"] + f"/node_{name}.npy")
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

        ax2.set_title("Consensus")

        fig2.savefig(figure_path + "consensus.png", dpi=300, bbox_inches="tight")

    else:
        raise ValueError("Invalid run type")
