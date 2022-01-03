from pathlib import Path
from datasets import MohlerDataset
import torch
from networks.graph_transformer import GraphTransformerNet
from tqdm import tqdm
from train import init
import yaml


class GraphInference:
    """
    Get model results from individual graphs.
    """

    def __init__(self, model_path: Path, config: dict) -> None:
        self.model_path = model_path
        self.config = config
        self.model = None

    def load_model(self) -> None:
        gt_model = GraphTransformerNet(self.config)
        checkpoint = torch.load(self.model_path)
        gt_model.load_state_dict(checkpoint['model_state_dict'])
        gt_model.to(self.config['device'])
        gt_model.eval()
        self.model = gt_model
        return


if __name__ == "__main__":

    config_file_path = Path("config.yaml")
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    config = init(config)

    model_path = Path('model/graphTrans_regress/model_epoch_12.pt')
