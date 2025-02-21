import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import toml
from typing import Dict, TypedDict
from numpy.typing import NDArray
from multiprocessing import Process
from gossip import create_gossip_network, Gossip


class NodeConfig(TypedDict):
    iterations: int
    step_size: float
    results_path: str


class Node(Process):
    configs: NodeConfig = None

    def __init__(
        self, id: str, comm: Gossip, initial_state: NDArray[np.float64]
    ):
        super().__init__()

        if self.configs is None:
            raise ValueError("Node configs not set")

        self.id = id
        self.comm = comm
        self.state = np.tile(initial_state, (self.configs["iterations"] + 1, 1))

    def run(self):
        for k in range(self.configs["iterations"]):
            self.comm.broadcast(self.state[k])
            neighbors_state = self.comm.gather()

            consensus_error = self.comm.degree * self.state[k] - sum(
                neighbors_state
            )

            self.state[k + 1] = (
                self.state[k] - self.configs["step_size"] * consensus_error
            )

        os.makedirs(self.configs["results_path"], exist_ok=True)
        np.save(self.configs["results_path"] + f"/node_{self.id}.npy", self.state)


if __name__ == "__main__":
    config = toml.load("configs/consensus.toml")

    node_ids = [nd["id"] for nd in config["NODE_DATA"]]
    edge_pairs = config["EDGE_PAIRS"]

    if config["RUN_TYPE"] == "ALG":
        Node.configs = config["NODE_CONFIGS"]

        nodes_state: Dict[str, NDArray[np.float64]] = {
            nd["id"]: np.array(nd["initial_state"]) for nd in config["NODE_DATA"]
        }
        gossip_network = create_gossip_network(node_ids, edge_pairs)

        consensus_nodes = [
            Node(id, gossip_network[id], nodes_state[id]) for id in node_ids
        ]

        for node in consensus_nodes:
            node.start()

        for node in consensus_nodes:
            node.join()

    elif config["RUN_TYPE"] == "VIS":
        figure_path = "figures/consensus/"

        fig1, ax1 = plt.subplots()

        node_pos = {nd["id"]: nd["position"] for nd in config["NODE_DATA"]}

        G = nx.Graph()
        G.add_nodes_from(node_ids)
        G.add_edges_from(edge_pairs)

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
            id: np.load(config["NODE_CONFIGS"]["results_path"] + f"/node_{id}.npy")
            for id in node_ids
        }

        for id in node_ids:
            for i in range(3):
                (line,) = ax2.plot(
                    states_dict[id][:, i],
                    color=node_colors[id],
                )

            line.set_label(f"Node {id}")

        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("State")

        ax2.legend(loc="upper right")

        ax2.set_title("Consensus")

        fig2.savefig(figure_path + "consensus.png", dpi=300, bbox_inches="tight")
