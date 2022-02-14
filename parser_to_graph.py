import traceback

from pandas import read_csv, read_excel, concat
from pathlib import Path
import spacy
import dgl
import torch
import numpy as np
from tqdm import tqdm
from gnnlens import Writer
from unidecode import unidecode


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
        self.df = read_excel(data_file)
        real_df = self.df[self.df['class'] == 'real']
        fake_df = self.df[self.df['class'] == 'fake']
        self.df = concat([real_df, fake_df])
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
        text = unidecode(text)
        spacy_doc = self.nlp(text)
        for token in spacy_doc:
            if token.has_vector:
                tid = self.nlp.vocab.vectors.find(key=token.norm)
                if tid == -1:
                    print("-1 ", token)
                    tid = self.oov_id
                node_token_id.append(tid)
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

        graph_dir = output_dir / 'Graph'
        graph_info_dir = output_dir / 'Graph_info'

        if not graph_dir.exists():
            graph_dir.mkdir(parents=True)

        if not graph_info_dir.exists():
            graph_info_dir.mkdir(parents=True)

        class_map = {'real': 0, 'fake': 1}
        for i, row in tqdm(self.df.iterrows(), desc='Processing data', total=self.df.shape[0]):
            try:
                data_id = row[self.columns[0]]
                graph_file_path = graph_dir / f'{data_id}.bin'
                graph_info_file_path = graph_info_dir / f'{data_id}.pkl'
                if graph_file_path.exists() and graph_info_file_path.exists():
                    continue
                raw_text = row[self.columns[1]]
                graph_rep = self.spacyParser(raw_text)
                if graph_rep['len'] < 9 or graph_rep['len'] > 1500:
                    continue
                label = row[self.columns[2]]
                dgl_graph = dgl.graph((graph_rep['src_nodes'], graph_rep['dst_nodes']), num_nodes=graph_rep['len'])
                dgl_graph.ndata['tokens'] = torch.from_numpy(np.array(graph_rep['node_token_id'], dtype=np.int16))
                dgl_graph.edata['type'] = torch.from_numpy(np.array(graph_rep['edge_type_id'], dtype=np.int8))
                node_texts = graph_rep['node_text']
                edge_types = graph_rep['edge_type']
                # labels = {'class': torch.tensor(class_map[label])}
                info_dict = {'graph_ids': data_id, 'texts': node_texts, 'dep_types': edge_types, 'class': label}
                dgl.save_graphs(graph_file_path.as_posix(), dgl_graph)
                dgl.data.utils.save_info(graph_info_file_path.as_posix(), info_dict)
            except:
                print(traceback.format_exc())
                print(f"Skipping Row: {i} - Text: {row[self.columns[1]]}")

        return True


if __name__ == '__main__':

    data_csv_file = Path("Dataset/Fake_News/v1/processed.xlsx")
    depedency_map_file = Path("assets/txt/dependency_tags.txt") # src = https://spacy.io/models/en#en_core_web_md-labels
    output_dir = Path("Dataset/Fake_News/v1/GT_graphs")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    SPG = SpacyParserGraph(data_csv_file, depedency_map_file, columns=['uid', 'text', 'class'])
    SPG.to_graph(output_dir)
