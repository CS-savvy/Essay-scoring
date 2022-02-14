from pathlib import Path
from datasets import MohlerDataset
import torch
from networks.graph_transformer import GraphTransformerNet
from train import init
import yaml
import dgl
import pandas as pd


class GraphInference:
    """
    Get model results from individual graphs.
    """

    def __init__(self, model_path: Path, config: dict) -> None:
        self.model = None
        self.model_path = model_path
        self.config = config
        self.load_model()
        self.class_map = {0: 'real', 1: 'fake'}

    def load_model(self) -> None:
        gt_model = GraphTransformerNet(**self.config)
        checkpoint = torch.load(self.model_path)
        gt_model.load_state_dict(checkpoint['model_state_dict'])
        gt_model.to(self.config['device'])
        gt_model.eval()
        self.model = gt_model
        return

    def load_dataset(self, graph_path: Path, info_path: Path) -> None:
        self.graphs, self.scores = dgl.load_graphs(graph_path.as_posix())
        self.scores = self.scores['score']
        self.n_samples = len(self.graphs)
        self.info = dgl.data.utils.load_info(info_path.as_posix())

    @staticmethod
    def load_graph(graph_path, graph_info_path):
        graph, _ = dgl.load_graphs(graph_path.as_posix())
        info = dgl.data.utils.load_info(graph_info_path.as_posix())
        return graph[0], info

    def prepare_batch(self, graph_id: str):
        index = self.info['graph_ids'].index(graph_id)
        texts = self.info['texts'][index]
        g = self.graphs[index]
        original_score = float(self.scores[index].numpy())
        g = MohlerDataset.laplacian_positional_encoding(g, self.config['pos_enc_dim'])
        batch_graphs = dgl.batch([g])
        batch_graphs = batch_graphs.to(self.config['device'])
        batch_x = batch_graphs.ndata['tokens'].to(self.config['device']).to(torch.long)  # num x feat
        batch_e = batch_graphs.edata['type'].to(self.config['device']).to(torch.long)
        batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(self.config['device'])
        sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(self.config['device'])
        sign_flip[sign_flip >= 0.5] = 1.0;
        sign_flip[sign_flip < 0.5] = -1.0
        batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        return batch_graphs, batch_x, batch_e, batch_lap_pos_enc, original_score, texts

    def prepare_for_infer(self, graph, info):
        texts = info['texts']
        original_class = info['class']
        g = MohlerDataset.laplacian_positional_encoding(graph, self.config['pos_enc_dim'])
        batch_graphs = dgl.batch([g])
        batch_graphs = batch_graphs.to(self.config['device'])
        batch_x = batch_graphs.ndata['tokens'].to(self.config['device']).to(torch.long)  # num x feat
        batch_e = batch_graphs.edata['type'].to(self.config['device']).to(torch.long)
        batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(self.config['device'])
        sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(self.config['device'])
        sign_flip[sign_flip >= 0.5] = 1.0;
        sign_flip[sign_flip < 0.5] = -1.0
        batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        return batch_graphs, batch_x, batch_e, batch_lap_pos_enc, original_class, texts

    # def infer(self, graph_id: str) -> tuple:
    #
    #     batch_graphs, batch_x, batch_e, batch_lap_pos_enc, original_score, texts = self.prepare_batch(graph_id)
    #     batch_scores = self.model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc)
    #     pred_score = float(batch_scores[0])
    #     return graph_id, str(original_score), str(pred_score),  " ".join(texts)

    # def infer_on_list(self, graph_ids: list, output_dir: Path) -> None:
    #     results = []
    #     for graph_id in tqdm(graph_ids, desc="Processing .."):
    #         result = self.infer(graph_id)
    #         results.append(result)
    #     df = pd.DataFrame(results, columns=["ID", "Original Score", "Predicted Score", "Original Text"])
    #     df.to_excel(output_dir / "inference_results.xlsx")
    #     return

    def infer(self, graph, graph_info):
        data = self.prepare_for_infer(graph, graph_info)
        batch_graphs, batch_x, batch_e, batch_lap_pos_enc, original_class, texts = data
        batch_scores = self.model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc)
        pred_score = torch.sigmoid(batch_scores)
        pred_index = int(torch.argmax(pred_score, dim=1)[0])
        pred_class = self.class_map[pred_index]
        return pred_class, original_class, texts

    def infer_on_list(self, graph_dir: Path, graph_info_dir: Path, graph_ids: list, output_dir: Path):
        results = []
        for graph_id in graph_ids:
            graph_path = graph_dir / (graph_id + '.bin')
            graph_info_path = graph_info_dir / (graph_id + '.pkl')
            graph, graph_info = GraphInference.load_graph(graph_path, graph_info_path)
            pred, target, texts = self.infer(graph, graph_info)
            results.append([graph_id, target, pred, " ".join(texts)])
        df = pd.DataFrame(results, columns=["ID", "Target", "Predicted", "Text"])
        df.to_excel(output_dir / "inference_results.xlsx")


if __name__ == "__main__":
    config_file_path = Path("config.yaml")
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    setup = init(config['Setup'])

    dataset_name = config['Setup']['dataset']
    dataset_config = config['Datasets'][dataset_name]
    mode = config['Setup']['mode']
    lap_pos_enc = config['Setup']['lap_pos_enc']

    model_config = config['Network']
    model_config['mode'] = mode
    model_config['lap_pos_enc'] = lap_pos_enc
    model_config['num_vocab'] = dataset_config['num_vocab']
    model_config['num_edge_type'] = dataset_config['num_edge_type']
    model_config['device'] = setup['device']
    model_config['n_class'] = dataset_config['n_class']

    graph_dir = Path('Dataset/Fake_News/v1/GT_graphs_10k/Graph')
    graph_info_dir = Path('Dataset/Fake_News/v1/GT_graphs_10k/Graph_info')
    model_path = Path('Experiments/FakeNews_init/model/model_epoch_4.pt')
    val_list_path = Path("Experiments/FakeNews_init/train.txt")
    output_dir = Path("Experiments/FakeNews_init/results")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    with open(val_list_path, 'r') as f:
        val_list = f.read().split("\n")
        val_list = [f for f in val_list if f]
    driver = GraphInference(model_path, model_config)
    driver.infer_on_list(graph_dir, graph_info_dir, val_list, output_dir)
