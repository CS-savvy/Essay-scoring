import torch
import dgl
from pathlib import Path
import spacy
from scipy import sparse as sp
import numpy as np


class MohlerDataset(torch.utils.data.Dataset):

    def __init__(self, graph_path: Path, graph_info_path: Path):
        """
        Loads DGL graph and its info into the memory

        :param graph_path: Path to the saved DGL graph.
        :param graph_info_path: Path to the label and info file.
        """

        self.graphs, self.labels = dgl.load_graphs(graph_path.as_posix())
        self.graphs = [g for g in self.graphs if 8 < g.num_nodes() < 50]
        self.scores = self.labels['score']
        self.n_samples = len(self.graphs)
        self.info = dgl.data.utils.load_info(graph_info_path.as_posix())
        self._collect_stats()

    def _collect_stats(self):
        self.max_nodes = max([g.num_nodes() for g in self.graphs])
        self.max_edges = max([g.num_edges() for g in self.graphs])

    @staticmethod
    def laplacian_positional_encoding(g, pos_enc_dim):
        """
            Graph positional encoding v/ Laplacian eigenvectors
        """

        # Laplacian
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(g.number_of_nodes()) - N * A * N

        # Eigenvectors with numpy
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort()  # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
        g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()

        return g

    def add_laplacian_positional_encodings(self, pos_enc_dim):
        # Graph positional encoding v/ Laplacian eigenvectors
        self.graphs = [MohlerDataset.laplacian_positional_encoding(g, pos_enc_dim) for g in self.graphs]

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graphs[idx], self.scores[idx]

    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.stack(labels).unsqueeze(dim=1).to(torch.long)
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels


if __name__ == '__main__':

    graph_path = Path('Dataset/mohler/GT_graphs/data.bin')
    graph_info_path = Path('Dataset/mohler/GT_graphs/data.pkl')
    dataset = MohlerDataset(graph_path, graph_info_path)

    print()