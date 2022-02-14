import torch
from captum.attr import IntegratedGradients
from pathlib import Path
from functools import partial
import yaml
from train import init
from tqdm import tqdm
from inference import GraphInference
import pandas as pd


class GraphAttributions:
    """
    Get model results from individual graphs.
    """

    def __init__(self, model_path: Path, config: dict, graph_dir: Path, graph_info_dir: Path) -> None:
        self.config = config
        self.driver = GraphInference(model_path, config)
        self.graph_dir = graph_dir
        self.graph_info_dir = graph_info_dir
        self.class_map = {0: 'real', 1: 'fake'}
        # self.driver.load_dataset(graph_path, graph_info_path)

    def get_attributions(self, graph_id: str) -> tuple:
        graph_file = self.graph_dir / (graph_id + ".bin")
        graph_info_file = self.graph_info_dir / (graph_id + ".pkl")
        graph, graph_info = GraphInference.load_graph(graph_file, graph_info_file)
        data = self.driver.prepare_for_infer(graph, graph_info)
        batch_graphs, batch_x, batch_e, batch_lap_pos_enc, original_score, texts = data
        batch_x_emb = self.driver.model.get_node_embedding(batch_x)
        batch_x_emb = batch_x_emb[None]
        score = self.driver.model.infer(batch_x_emb, batch_e, batch_graphs, batch_lap_pos_enc)
        pred_score = torch.sigmoid(score)
        pred_index = int(torch.argmax(pred_score, dim=1)[0])
        baseline_batch_x = torch.zeros_like(batch_x_emb).to(self.config['device'])
        ig = IntegratedGradients(partial(self.driver.model.infer, e=batch_e, g=batch_graphs, h_lap_pos_enc=batch_lap_pos_enc))
        attributions, delta = ig.attribute(batch_x_emb, baseline_batch_x, target=pred_index, return_convergence_delta=True)
        attributions = attributions.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()

        return original_score, self.class_map[pred_index], attributions, texts, delta

    def exec_on_list(self, graph_ids: list, output_dir: Path) -> None:
        results = []
        for graph_id in tqdm(graph_ids, desc="Processing .."):
            original_score, pred_score, attributions, texts, delta = self.get_attributions(graph_id)
            text_weights = [f"{t} ({w : .4f})" for t, w in zip(texts, attributions)]
            results.append((graph_id, str(original_score), str(pred_score),  " ".join(text_weights)))

        df = pd.DataFrame(results, columns=["ID", "Original Score", "Predicted Score", "Attributions"])
        df.to_excel(output_dir / "attribution_results.xlsx")


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
    with open(val_list_path, 'r') as f:
        val_list = f.read().split("\n")

    attribution_driver = GraphAttributions(model_path, model_config, graph_dir, graph_info_dir)
    attribution_driver.exec_on_list(val_list, output_dir)
