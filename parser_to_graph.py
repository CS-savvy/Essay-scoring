from pandas import read_csv
from pathlib import Path
import spacy
import dgl
import torch
import numpy as np
from tqdm import tqdm


class SpacyParserGraph:
    """
    Use spacy dependency parser to create DGL graph objects

    """

    def __init__(self, data_file: Path, dependency_map_file: Path, columns: list):
        """
        Loads texts dataframe and dependency type list.

        :param data_file: Path of data csv file which contains texts.
        :param dependency_map_file: path of file path which enumerates dependency type.
        :param columns: A list of useful columns in the order of ID, texts, label.
        """
        self.df = read_csv(data_file, delimiter=",", encoding='utf8')
        with open(dependency_map_file, 'r') as f:
            self.dep_map = f.read().split("\n")
        print("Loading spacy model .. ")
        self.nlp = spacy.load('en_core_web_md')
        self.columns = columns
        self.oov_id = 20000

    def spacyParser(self, text: str, verbose: bool = False) -> dict:
        """
        Applies spacy dependency parser on text to generate graph representation

        :param text: raw text (sentence or paragraph)
        :param verbose: whether to print parser output.
        :return: graph representation and node/edge attributes
        """
        src_nodes = []
        dst_nodes = []
        edge_type = []
        edge_type_id = []
        node_token_id = []
        node_text = []

        spacy_doc = self.nlp(text)
        for token in spacy_doc:
            if token.has_vector:
                node_token_id.append(self.nlp.vocab.vectors.find(key=token.norm))
            else:
                node_token_id.append(self.oov_id)
            node_text.append(token.text)
            if spacy.explain(token.dep_) is None:
                # print(token, spacy.explain(token.dep_))
                continue
            src_nodes.append(token.head.i)
            dst_nodes.append(token.i)
            edge_type_id.append(self.dep_map.index(token.dep_))
            edge_type.append(token.dep_)

            if verbose:
                print(f" \n {token.text:{8}} {token.i} {token.dep_ + ' =>':{10}} "
                      f" {token.head.text:{9}} {token.head.i}"
                      f" {spacy.explain(token.dep_)} ")

        return {'src_nodes': src_nodes, 'dst_nodes': dst_nodes, 'edge_type': edge_type, 'len': len(spacy_doc),
                'edge_type_id': edge_type_id, 'node_token_id': node_token_id, 'node_text': node_text}

    def to_graph(self, output_dir: Path, shards: int = 10) -> bool:
        """
        converts raw text to graph and save it in defined output directory.

        :param output_dir: directory where graphs and labels will be saved.
        :param shards: distribute graphs in n number of files.
        :return: status
        """
        # TODO: distribute graphs in multiple files
        graphs = []
        graph_id = []
        graph_label = []
        node_texts = []
        edge_types = []
        for i, row in tqdm(self.df.iterrows(), desc='Processing data', total=self.df.shape[0]):
            raw_text = row[self.columns[1]]
            graph_rep = self.spacyParser(raw_text)
            if graph_rep['len'] < 9 or graph_rep['len'] > 64:
                continue
            score = row[self.columns[2]]
            if score > 5:
                continue
            graph_label.append(score)
            dgl_graph = dgl.graph((graph_rep['src_nodes'], graph_rep['dst_nodes']), num_nodes=graph_rep['len'])
            dgl_graph.ndata['tokens'] = torch.from_numpy(np.array(graph_rep['node_token_id'], dtype=np.int16))
            dgl_graph.edata['type'] = torch.from_numpy(np.array(graph_rep['edge_type_id'], dtype=np.int8))
            graphs.append(dgl_graph)
            graph_id.append(row[self.columns[0]])
            node_texts.append(graph_rep['node_text'])
            edge_types.append(graph_rep['edge_type'])

        labels = {'score': torch.from_numpy(np.array(graph_label, dtype=np.float16))}
        info_dict = {'graph_ids': graph_id, 'texts': node_texts, 'dep_types': edge_types}
        graph_file_path = output_dir / 'data.bin'
        graph_info_file_path = output_dir / 'data.pkl'

        # save files
        dgl.save_graphs(graph_file_path.as_posix(), graphs, labels=labels)
        dgl.data.utils.save_info(graph_info_file_path.as_posix(), info_dict)
        return True


if __name__ == '__main__':

    data_csv_file = Path("Dataset/mohler/mohler_processed.csv")
    depedency_map_file = Path("assets/txt/dependency_tags.txt") # src = https://spacy.io/models/en#en_core_web_md-labels
    output_dir = Path("Dataset/mohler/GT_graphs")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    SPG = SpacyParserGraph(data_csv_file, depedency_map_file, columns=['uid', 'student_answer', 'score_avg'])
    SPG.to_graph(output_dir)
