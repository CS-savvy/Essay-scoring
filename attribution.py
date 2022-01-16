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

    def __init__(self, model_path: Path, config: dict, graph_path: Path, graph_info_path: Path) -> None:
        self.config = config
        self.driver = GraphInference(model_path, config)
        self.driver.load_dataset(graph_path, graph_info_path)

    def get_attributions(self, graph_id: str) -> tuple:
        data = self.driver.prepare_batch(graph_id)
        batch_graphs, batch_x, batch_e, batch_lap_pos_enc, original_score, texts = data
        batch_x_emb = self.driver.model.get_node_embedding(batch_x)
        batch_x_emb = batch_x_emb[None]
        score = self.driver.model.infer(batch_x_emb, batch_e, batch_graphs, batch_lap_pos_enc)
        pred_score = float(score[0])
        baseline_batch_x = torch.zeros_like(batch_x_emb).to(self.config['device'])
        ig = IntegratedGradients(partial(self.driver.model.infer, e=batch_e, g=batch_graphs, h_lap_pos_enc=batch_lap_pos_enc))
        attributions, delta = ig.attribute(batch_x_emb, baseline_batch_x, target=0, return_convergence_delta=True)
        attributions = attributions.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()

        return original_score, pred_score, attributions, texts, delta

    def exec_on_list(self, graph_ids: list, output_dir: Path) -> None:
        results = []
        for graph_id in tqdm(graph_ids, desc="Processing .."):
            original_score, pred_score, attributions, texts, delta = self.get_attributions(graph_id)
            text_weights = [f"{t} ({w : .4f})" for t, w in zip(texts, attributions)]
            results.append((graph_id, str(original_score), str(pred_score),  " ".join(text_weights)))

        df = pd.DataFrame(results, columns=["ID", "Original Score", "Predicted Score", "Attributions"])
        df.to_excel(output_dir / "inference_results.xlsx")


if __name__ == "__main__":

    config_file_path = Path("config.yaml")
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    config = init(config)
    graph_path = Path('Dataset/mohler/GT_graphs/data.bin')
    graph_info_path = Path('Dataset/mohler/GT_graphs/data.pkl')
    model_path = Path('model/with_word_spacy_embed/model_epoch_248.pt')
    val_list_path = Path("model/graphTrans_regress/train.txt")
    output_dir = Path("model/with_word_spacy_embed/")
    with open(val_list_path, 'r') as f:
        val_list = f.read().split("\n")
    attribution_driver = GraphAttributions(model_path, config, graph_path, graph_info_path)
    attribution_driver.exec_on_list(val_list, output_dir)
