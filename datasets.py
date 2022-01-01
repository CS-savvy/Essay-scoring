import torch
import dgl
from pathlib import Path
import spacy
import numpy as np


class MohlerDataset(torch.utils.data.Dataset):

    def __init__(self, graph_path: Path, graph_info_path: Path):
        """
        Loads DGL graph and its info into the memory

        :param graph_path: Path to the saved DGL graph.
        :param graph_info_path: Path to the label and info file.
        """

        self.graphs, self.labels = dgl.load_graphs(graph_path.as_posix())
        self.n_samples = len(self.graphs)
        self.info = dgl.data.utils.load_info(graph_info_path.as_posix())
        print()

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
        return self.graph_lists[idx], self.graph_labels[idx]

    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels


if __name__ == '__main__':

    graph_path = Path('Dataset/mohler/GT_graphs/data.bin')
    graph_info_path = Path('Dataset/mohler/GT_graphs/data.pkl')
    dataset = MohlerDataset(graph_path, graph_info_path)

    print()