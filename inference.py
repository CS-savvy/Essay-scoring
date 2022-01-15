from pathlib import Path
from datasets import MohlerDataset
import torch
from networks.graph_transformer import GraphTransformerNet
from tqdm import tqdm
from train import init
import yaml
import dgl
import pandas as pd


class GraphInference:
    """
    Get model results from individual graphs.
    """

    def __init__(self, model_path: Path, config: dict) -> None:
        self.graphs = None
        self.model = None
        self.info = None
        self.n_samples = None
        self.scores = None
        self.model_path = model_path
        self.config = config
        self.load_model()

    def load_model(self) -> None:
        gt_model = GraphTransformerNet(self.config)
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

    def infer(self, graph_id: str) -> tuple:

        batch_graphs, batch_x, batch_e, batch_lap_pos_enc, original_score, texts = self.prepare_batch(graph_id)
        batch_scores = self.model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc)
        pred_score = float(batch_scores[0])
        return graph_id, str(original_score), str(pred_score),  " ".join(texts)

    def infer_on_list(self, graph_ids: list, output_dir: Path) -> None:
        results = []
        for graph_id in tqdm(graph_ids, desc="Processing .."):
            result = self.infer(graph_id)
            results.append(result)
        df = pd.DataFrame(results, columns=["ID", "Original Score", "Predicted Score", "Original Text"])
        df.to_excel(output_dir / "inference_results.xlsx")
        return


if __name__ == "__main__":

    config_file_path = Path("config.yaml")
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    config = init(config)
    graph_path = Path('Dataset/mohler/GT_graphs/data.bin')
    graph_info_path = Path('Dataset/mohler/GT_graphs/data.pkl')
    model_path = Path('model/with_word_spacy_embed/model_epoch_248.pt')
    val_list_path = Path("model/graphTrans_regress/val.txt")
    output_path = Path("model/with_word_spacy_embed/")
    with open(val_list_path, 'r') as f:
        val_list = f.read().split("\n")
    driver = GraphInference(model_path, config)
    driver.load_dataset(graph_path, graph_info_path)
    driver.infer(val_list, output_path)
