import math
import numpy as np
import networkx as nx
from scipy import sparse
from tqdm import tqdm


class QAOA_FEATHER:
    """
    Graph-level FEATHER embedding specialized for QAOA / circuit graphs.

    - No sklearn
    - No karateclub
    - Deterministic
    - Uses weighted adjacency
    """

    def __init__(
        self,
        theta_max: float = 2.5,
        eval_points: int = 16,
        order: int = 5,
        pooling: str = "mean",
    ):
        self.theta_max = theta_max
        self.eval_points = eval_points
        self.order = order
        self.pooling = pooling

    # ------------------------------------------------------------------
    # Graph utilities
    # ------------------------------------------------------------------
    def _normalized_adjacency(self, graph: nx.Graph):
        """
        Compute D^{-1} A for a weighted graph.
        """
        n = graph.number_of_nodes()
        A = nx.to_scipy_sparse_array(
            graph,
            nodelist=range(n),
            weight="weight",
            format="coo",
        )

        degrees = np.array(A.sum(axis=1)).flatten()
        degrees[degrees == 0.0] = 1.0  # avoid division by zero

        D_inv = sparse.diags(1.0 / degrees)
        return D_inv @ A

    def _node_features(self, graph: nx.Graph):
        """
        Node features tuned for QAOA graphs.
        """
        n = graph.number_of_nodes()

        # log-degree (weighted)
        weighted_deg = np.array(
            [graph.degree(v, weight="weight") for v in range(n)]
        )
        log_deg = np.log(weighted_deg + 1.0).reshape(-1, 1)

        # clustering coefficient (structure)
        clust = np.array(
            [nx.clustering(graph, v, weight="weight") for v in range(n)]
        ).reshape(-1, 1)

        return np.concatenate([log_deg, clust], axis=1)

    # ------------------------------------------------------------------
    # FEATHER core
    # ------------------------------------------------------------------
    def _feather_node_embedding(self, graph: nx.Graph):
        """
        Compute node-level FEATHER embeddings.
        """
        n = graph.number_of_nodes()
        assert sorted(graph.nodes) == list(range(n)), \
            "Nodes must be indexed 0..N-1"

        A_tilde = self._normalized_adjacency(graph)

        X0 = self._node_features(graph)
        theta = np.linspace(0.01, self.theta_max, self.eval_points)

        # Characteristic functions
        X = np.einsum("nd,t->ndt", X0, theta).reshape(n, -1)
        X = np.concatenate([np.cos(X), np.sin(X)], axis=1)

        blocks = []
        for _ in range(self.order):
            X = A_tilde @ X
            blocks.append(X)

        return np.concatenate(blocks, axis=1)

    def _pool(self, node_features: np.ndarray):
        if self.pooling == "max":
            return node_features.max(axis=0)
        elif self.pooling == "min":
            return node_features.min(axis=0)
        return node_features.mean(axis=0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed_graph(self, graph: nx.Graph):
        """
        Compute a single graph embedding.
        """
        node_emb = self._feather_node_embedding(graph)
        return self._pool(node_emb)

    def fit(self, graphs):
        """
        Embed a list of graphs.
        """
        self.embeddings_ = np.array(
            [self.embed_graph(g) for g in tqdm(graphs)],
            dtype=np.float32
        )
        return self

    def get_embedding(self):
        return self.embeddings_
