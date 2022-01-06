import torch
import dgl
from captum.attr import IntegratedGradients
from pathlib import Path
from functools import partial
import yaml
from train import init
from datasets import MohlerDataset
from networks.graph_transformer import GraphTransformerNet


class GraphAttributions:
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

    def get_attributions(self, graph_id):
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
        batch_x_emb = self.model.get_node_embedding(batch_x)
        batch_x_emb = batch_x_emb[None]
        score = self.model.infer(batch_x_emb, batch_e, batch_graphs, batch_lap_pos_enc)
        baseline_batch_x = torch.zeros_like(batch_x_emb).to(config['device'])
        ig = IntegratedGradients(partial(self.model.infer, e=batch_e, g=batch_graphs, h_lap_pos_enc=batch_lap_pos_enc))
        attributions, delta = ig.attribute(batch_x_emb, baseline_batch_x, target=0, return_convergence_delta=True)
        print('IG Attributions:', attributions)
        print('Convergence Delta:', delta)

s
if __name__ == "__main__":

    config_file_path = Path("config.yaml")
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    config = init(config)
    graph_path = Path('Dataset/mohler/GT_graphs/data.bin')
    graph_info_path = Path('Dataset/mohler/GT_graphs/data.pkl')
    model_path = Path('model/with_word_spacy_embed/model_epoch_248.pt')
    val_list_path = Path("model/graphTrans_regress/val.txt")
    output_dir = Path("model/with_word_spacy_embed/")
    with open(val_list_path, 'r') as f:
        val_list = f.read().split("\n")
    driver = GraphAttributions(model_path, config)
    driver.load_dataset(graph_path, graph_info_path)
    driver.get_attributions(val_list, output_dir)