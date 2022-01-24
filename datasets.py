import torch
import dgl
from pathlib import Path
from scipy import sparse as sp
import numpy as np
from collections import Counter


class MohlerDataset(torch.utils.data.Dataset):

    def __init__(self, graph_path: Path, graph_info_path: Path, type: str = 'regression'):
        """
        Loads DGL graph and its info into the memory

        :param graph_path: Path to the saved DGL graph.
        :param graph_info_path: Path to the label and info file.
        """

        self.graphs, self.scores = dgl.load_graphs(graph_path.as_posix())
        self.scores = self.scores['score'].type(torch.float32)
        self.type = type
        self.vocab = None
        self.dependency_list = None
        if type == 'classification':
            self.labels = torch.round(self.scores * 2) / 2.0
            self.class_map = {s: c for s, c in zip(np.arange(0, 5.5, 0.5), np.arange(0, 11))}
            self.inverse_class_map = {val: key for key, val in self.class_map.items()}
            self.labels = torch.tensor([self.class_map[float(s)] for s in self.labels], dtype=torch.float32)
        self.n_samples = len(self.graphs)
        self.info = dgl.data.utils.load_info(graph_info_path.as_posix())
        self._collect_stats()
        self._update_tokens()

    def _update_tokens(self):
        for g, text, dep in zip(self.graphs, self.info['texts'], self.info['dep_types']):
            word_tokens = [self.vocab.index(t) if t in self.vocab else 1999 for t in text]
            dep_tokens = [self.dependency_list.index(d) if d in self.dependency_list else 29 for d in dep]
            word_tokens = torch.tensor(word_tokens).type(torch.int64)
            dep_tokens = torch.tensor(dep_tokens).type(torch.int64)
            g.ndata['tokens'] = word_tokens
            g.edata['type'] = dep_tokens

    def _collect_stats(self):
        self.max_nodes = max([g.num_nodes() for g in self.graphs])
        self.max_edges = max([g.num_edges() for g in self.graphs])
        vocab_count = Counter()
        for k in self.info['texts']:
            k = [s.lower() for s in k]
            vocab_count.update(k)
        vocab_count = [k for k in vocab_count.items()]
        vocab_count = sorted(vocab_count, key=lambda x: x[1], reverse=True)
        final_vocab = [k[0] for k in vocab_count if k[1] > 1]
        low_freq_words = [k[0] for k in vocab_count if k[1] == 1]
        low_freq_words = sorted(low_freq_words, key=lambda x: len(x), reverse=True)
        final_vocab += low_freq_words[:642]
        self.vocab = final_vocab
        dep_count = Counter()
        for k in self.info['dep_types']:
            dep_count.update(k)
        dep_count = [k for k in dep_count.items()]
        dep_count = sorted(dep_count, key=lambda x: x[1], reverse=True)
        final_dep_type = [k[0] for k in dep_count][:29]
        self.dependency_list = final_dep_type

    def class_to_score(self, class_id):
        return float(self.inverse_class_map[class_id])

    def get_graph_ids(self, indices):
        return [self.info['graph_ids'][i] for i in indices]

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
        if self.type == 'classification':
            return self.graphs[idx], self.labels[idx]
        return self.graphs[idx], self.scores[idx]

    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        if self.type == 'classification':
            labels = torch.stack(labels).type(torch.long)
        else:
            labels = torch.stack(labels).unsqueeze(dim=1).type(torch.float32)
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels


class SbankDataset(torch.utils.data.Dataset):

    def __init__(self, graph_path: Path, graph_info_path: Path, type: str = 'regression'):
        """
        Loads DGL graph and its info into the memory

        :param graph_path: Path to the saved DGL graph.
        :param graph_info_path: Path to the label and info file.
        """

        self.graphs, self.scores = dgl.load_graphs(graph_path.as_posix())
        self.scores = self.scores['score'].type(torch.float32)
        self.type = type
        self.vocab = None
        self.dependency_list = None
        if type == 'classification':
            self.class_map = {s: c for s, c in zip(np.arange(0, 5), np.arange(0, 5))}
            self.inverse_class_map = {val: key for key, val in self.class_map.items()}
            self.labels = torch.tensor([self.class_map[float(s)] for s in self.scores], dtype=torch.float32)
        self.n_samples = len(self.graphs)
        self.info = dgl.data.utils.load_info(graph_info_path.as_posix())
        self._collect_stats()
        self._update_tokens()

    def _update_tokens(self):
        for g, text, dep in zip(self.graphs, self.info['texts'], self.info['dep_types']):
            word_tokens = [self.vocab.index(t) if t in self.vocab else 1999 for t in text]
            dep_tokens = [self.dependency_list.index(d) if d in self.dependency_list else 29 for d in dep]
            word_tokens = torch.tensor(word_tokens).type(torch.int64)
            dep_tokens = torch.tensor(dep_tokens).type(torch.int64)
            g.ndata['tokens'] = word_tokens
            g.edata['type'] = dep_tokens

    def _collect_stats(self):
        self.max_nodes = max([g.num_nodes() for g in self.graphs])
        self.max_edges = max([g.num_edges() for g in self.graphs])
        vocab_count = Counter()
        for k in self.info['texts']:
            k = [s.lower() for s in k]
            vocab_count.update(k)
        vocab_count = [k for k in vocab_count.items()]
        vocab_count = sorted(vocab_count, key=lambda x: x[1], reverse=True)
        final_vocab = [k[0] for k in vocab_count if k[1] > 1]
        low_freq_words = [k[0] for k in vocab_count if k[1] == 1]
        low_freq_words = sorted(low_freq_words, key=lambda x: len(x), reverse=True)
        final_vocab += low_freq_words[:457]
        self.vocab = final_vocab
        dep_count = Counter()
        for k in self.info['dep_types']:
            dep_count.update(k)
        dep_count = [k for k in dep_count.items()]
        dep_count = sorted(dep_count, key=lambda x: x[1], reverse=True)
        final_dep_type = [k[0] for k in dep_count][:29]
        self.dependency_list = final_dep_type

    def class_to_score(self, class_id):
        return self.inverse_class_map[class_id]

    def get_graph_ids(self, indices):
        return [self.info['graph_ids'][i] for i in indices]

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
        if self.type == 'classification':
            return self.graphs[idx], self.labels[idx]
        return self.graphs[idx], self.scores[idx]

    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        if self.type == 'classification':
            labels = torch.stack(labels).type(torch.long)
        else:
            labels = torch.stack(labels).unsqueeze(dim=1).type(torch.float32)
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels


if __name__ == '__main__':

    # graph_path = Path('Dataset/mohler/GT_graphs/data.bin')
    # graph_info_path = Path('Dataset/mohler/GT_graphs/data.pkl')
    # dataset = MohlerDataset(graph_path, graph_info_path, type='classification')

    graph_path = Path('Dataset/scientsbank/GT_graphs/data.bin')
    graph_info_path = Path('Dataset/scientsbank/GT_graphs/data.pkl')
    dataset = SbankDataset(graph_path, graph_info_path, type='classification')
    print()