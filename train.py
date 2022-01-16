import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim as optim
import yaml
import time
from pathlib import Path
from datasets import MohlerDataset
import random
import numpy as np
from networks.graph_transformer import GraphTransformerNet
from tqdm import tqdm
from metrics import MAE, MSE
from torch.utils.tensorboard import SummaryWriter
from eval import evaluate_network
import spacy


def train_epoch(model: nn.Module, optimizer: optim, device: torch.device, data_loader: DataLoader, logger: tqdm) -> tuple:
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    epoch_train_mse = 0
    for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
        logger.set_postfix({'batch': iter})
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['tokens'].to(device).to(torch.long)  # num x feat
        batch_e = batch_graphs.edata['type'].to(device).to(torch.long)
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()
        try:
            batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
            sign_flip[sign_flip >= 0.5] = 1.0;
            sign_flip[sign_flip < 0.5] = -1.0
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        except:
            batch_lap_pos_enc = None

        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc)
        loss = model.loss(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(batch_scores, batch_targets)
        epoch_train_mse += MSE(batch_scores, batch_targets)

    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    epoch_train_mse /= (iter + 1)

    return epoch_loss, epoch_train_mae, epoch_train_mse, optimizer


def train_model(model: object, dataset: object, config: dict, verbose: bool = True) -> None:
    """
    Train GraphTransformer Network
    :param model:
    :param dataset:
    :param config:
    :param epoch:
    :param verbose:
    :return:
    """
    dataset_size = len(dataset)
    split_point = int(np.floor(dataset_size*(1-config['val_split'])))
    indices = list(range(dataset_size))

    if config['shuffle']:
        random.shuffle(indices)
    train_sampler = SubsetRandomSampler(indices[:split_point])
    val_sampler = SubsetRandomSampler(indices[split_point:])

    train_graph_ids = dataset.get_graph_ids(train_sampler.indices)
    with open(config['model_dir'] / 'train.txt', 'w') as f:
        f.write("\n".join(train_graph_ids))

    val_graph_ids = dataset.get_graph_ids(val_sampler.indices)
    with open(config['model_dir'] / 'val.txt', 'w') as f:
        f.write("\n".join(val_graph_ids))

    train_loader = DataLoader(dataset, batch_size=config['batch_size'],
                              sampler=train_sampler, collate_fn=dataset.collate)
    val_loader = DataLoader(dataset, batch_size=config['batch_size'],
                            sampler=val_sampler, collate_fn=dataset.collate)
    model = model.to(config['device'])

    optimizer = optim.Adam(model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=config['lr_reduce_factor'],
                                                     patience=config['lr_schedule_patience'],
                                                     verbose=True)

    writer = SummaryWriter(log_dir=config['model_dir'])

    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_MAEs, epoch_val_MAEs = [], []
    epoch_train_MSEs, epoch_val_MSEs = [], []

    min_val_mae = 1000000
    with tqdm(range(config['epochs'])) as t:
        for epoch in t:
            t.set_description('Epoch %d' % epoch)
            start = time.time()
            epoch_train_loss, epoch_train_mae, epoch_train_mse, optimizer = train_epoch(model, optimizer, config['device'], train_loader, t)
            epoch_val_loss, epoch_val_mae, epoch_val_mse = evaluate_network(model, config['device'], val_loader)

            epoch_train_losses.append(epoch_train_loss)
            epoch_val_losses.append(epoch_val_loss)
            epoch_train_MAEs.append(epoch_train_mae)
            epoch_train_MSEs.append(epoch_train_mse)
            epoch_val_MAEs.append(epoch_val_mae)
            epoch_val_MSEs.append(epoch_val_mse)

            print(f"\n Time: {time.time()-start} | lr: {optimizer.param_groups[0]['lr']} | train loss: {epoch_train_loss}"
                  f" val loss: {epoch_val_loss} | train MAE: {epoch_train_mae} | val MAE: {epoch_val_mae}")
            scheduler.step(epoch_val_loss)

            writer.add_scalar('train/_loss', epoch_train_loss, epoch)
            writer.add_scalar('val/_loss', epoch_val_loss, epoch)
            writer.add_scalar('train/_mae', epoch_train_mae, epoch)
            writer.add_scalar('train/_mse', epoch_train_mse, epoch)
            writer.add_scalar('val/_mae', epoch_val_mae, epoch)
            writer.add_scalar('val/_mse', epoch_val_mse, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

            if epoch > 3 and min_val_mae > epoch_val_mae:
                print("Saving model ....")
                min_val_mae = epoch_val_mae
                model_content = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                                 'val_mae': epoch_val_mae, 'train_mae': epoch_train_mae}
                torch.save(model_content, config['model_dir'] / f"model_epoch_{epoch}.pt")
    writer.close()


def init(configs: dict) -> dict:
    if configs['device'] in ['auto', 'gpu']:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("GPU not available, using cpu only.")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    configs['device'] = device

    random.seed(configs['seed'])
    np.random.seed(configs['seed'])
    torch.manual_seed(configs['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(configs['seed'])

    model_dir = Path(configs['model_dir'])
    if not model_dir.exists():
        model_dir.mkdir(parents=True)
    else:
        print("Model dir already exists")
        exit()
    configs['model_dir'] = model_dir
    return configs


def get_embedding_weights():
    nlp = spacy.load('en_core_web_md')
    embeddings = nlp.vocab.vectors.data
    oov_vector = np.zeros((1, 300))
    final_embedding = np.vstack([embeddings, oov_vector])
    return final_embedding


if __name__ == "__main__":

    config_file_path = Path("config.yaml")
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    config = init(config)

    graph_path = Path(config['graph_file'])
    graph_info_path = Path(config['graph_info_file'])
    mohler_dataset = MohlerDataset(graph_path, graph_info_path)
    if config['lap_pos_enc']:
        mohler_dataset.add_laplacian_positional_encodings(config['pos_enc_dim'])
    # print("Getting Spacy embeddings .. ")
    # word_embeddings = get_embedding_weights()
    print("Done.")
    # word_embeddings = torch.from_numpy(word_embeddings.astype(np.float32))
    gt_model = GraphTransformerNet(config)
    train_model(gt_model, mohler_dataset, config)

