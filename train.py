import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim as optim
import yaml
import time
from pathlib import Path
from datasets import MohlerDataset, SbankDataset, FakeNews
import random
import numpy as np
from networks.graph_transformer import GraphTransformerNet
from tqdm import tqdm
from metrics import MAE, MSE
from torch.utils.tensorboard import SummaryWriter
import spacy
from time import sleep


def train_epoch(model: nn.Module, optimizer: optim, device: torch.device, data_loader: DataLoader, logger: tqdm) -> tuple:
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    epoch_train_mse = 0
    epoch_train_accuracy = 0
    for iter, (batch_graphs, batch_targets, graph_ids) in enumerate(data_loader):
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

        batch_scores = model.forward(batch_graphs, batch_x, batch_e, graph_ids, batch_lap_pos_enc)
        loss = model.loss(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()

        if mode == 'classification':
            pred_score = torch.sigmoid(batch_scores)
            pred_class = torch.argmax(pred_score, dim=1)
            pred_label = torch.tensor([dataset.class_to_score(r) for r in pred_class.cpu().numpy()], dtype=torch.float32)
            target_label = torch.tensor([dataset.class_to_score(r) for r in batch_targets.cpu().numpy()], dtype=torch.float32)
            epoch_train_mae += MAE(pred_label, target_label)
            epoch_train_mse += MSE(pred_label, target_label)
            epoch_train_accuracy += torch.sum(target_label == pred_label).item() / len(target_label)
        else:
            epoch_train_mae += MAE(batch_scores, batch_targets)
            epoch_train_mse += MSE(batch_scores, batch_targets)

    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    epoch_train_mse /= (iter + 1)
    metrics = {'epoch_train_loss': epoch_loss, 'epoch_train_mae': epoch_train_mae, 'epoch_train_mse': epoch_train_mse}
    if mode == 'classification':
        epoch_train_accuracy /= (iter + 1)
        metrics['epoch_train_accuracy'] = epoch_train_accuracy

    return metrics, optimizer


def evaluate_network(model, device, data_loader):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    epoch_test_mse = 0
    epoch_test_accuracy = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets, graph_ids) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['tokens'].to(device).to(torch.long)  # num x feat
            batch_e = batch_graphs.edata['type'].to(device).to(torch.long)
            batch_targets = batch_targets.to(device)
            try:
                batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            except:
                batch_lap_pos_enc = None

            batch_scores = model.forward(batch_graphs, batch_x, batch_e, graph_ids, batch_lap_pos_enc)
            loss = model.loss(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()

            if mode == 'classification':
                pred_score = torch.sigmoid(batch_scores)
                pred_class = torch.argmax(pred_score, dim=1)
                pred_label = torch.tensor([dataset.class_to_score(r) for r in pred_class.cpu().numpy()], dtype=torch.float32)
                target_label = torch.tensor([dataset.class_to_score(r) for r in batch_targets.cpu().numpy()], dtype=torch.float32)
                epoch_test_mae += MAE(pred_label, target_label)
                epoch_test_mse += MSE(pred_label, target_label)
                epoch_test_accuracy += torch.sum(target_label == pred_label).item() / len(target_label)
            else:
                epoch_test_mae += MAE(batch_scores, batch_targets)
                epoch_test_mse += MSE(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)

        epoch_test_mse /= (iter + 1)
        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
        metrics = {'epoch_test_loss': epoch_test_loss, 'epoch_test_mae': epoch_test_mae,
                   'epoch_test_mse': epoch_test_mse}
        if mode == 'classification':
            epoch_test_accuracy /= (iter + 1)
            metrics['epoch_test_accuracy'] = epoch_test_accuracy

    return metrics


def train_model(model: object, dataset: object, mode: str,
                device, val_split=0.2, **kwargs) -> None:
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
    split_point = int(np.floor(dataset_size*(1-val_split)))
    indices = list(range(dataset_size))

    if kwargs['shuffle']:
        random.shuffle(indices)
    train_sampler = SubsetRandomSampler(indices[:split_point])
    val_sampler = SubsetRandomSampler(indices[split_point:])

    train_graph_ids = dataset.get_graph_ids(train_sampler.indices)
    with open(kwargs['exp_dir'] / 'train.txt', 'w') as f:
        f.write("\n".join(train_graph_ids))

    val_graph_ids = dataset.get_graph_ids(val_sampler.indices)
    with open(kwargs['exp_dir'] / 'val.txt', 'w') as f:
        f.write("\n".join(val_graph_ids))

    train_loader = DataLoader(dataset, batch_size=kwargs['batch_size'],
                              sampler=train_sampler, collate_fn=dataset.collate)
    val_loader = DataLoader(dataset, batch_size=kwargs['batch_size'],
                            sampler=val_sampler, collate_fn=dataset.collate)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=kwargs['init_lr'], weight_decay=kwargs['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=kwargs['lr_reduce_factor'],
                                                     patience=kwargs['lr_schedule_patience'],
                                                     verbose=True)

    writer = SummaryWriter(log_dir=kwargs['exp_dir'] / 'events')

    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_MAEs, epoch_val_MAEs = [], []
    epoch_train_MSEs, epoch_val_MSEs = [], []
    epoch_train_ACCs, epoch_val_ACCs = [], []

    min_val_mae = 1000000
    with tqdm(range(kwargs['epochs'])) as t:
        for epoch in t:
            t.set_description('Epoch %d' % epoch)
            start = time.time()

            train_metric, optimizer = train_epoch(model, optimizer, device, train_loader, t)
            val_metric = evaluate_network(model, device, val_loader)

            epoch_train_losses.append(train_metric['epoch_train_loss'])
            epoch_val_losses.append(val_metric['epoch_test_loss'])
            epoch_train_MAEs.append(train_metric['epoch_train_mae'])
            epoch_val_MAEs.append(val_metric['epoch_test_mae'])
            epoch_train_MSEs.append(train_metric['epoch_train_mse'])
            epoch_val_MSEs.append(val_metric['epoch_test_mse'])

            if mode == 'classification':
                epoch_train_ACCs.append(train_metric['epoch_train_accuracy'])
                epoch_val_ACCs.append(val_metric['epoch_test_accuracy'])


            print(f"\n Time: {time.time()-start} | lr: {optimizer.param_groups[0]['lr']} | train loss: {train_metric['epoch_train_loss']}"
                  f" val loss: {val_metric['epoch_test_loss']} | train MAE: {train_metric['epoch_train_mae']} | val MAE: {val_metric['epoch_test_mae']}")
            scheduler.step(val_metric['epoch_test_loss'])

            writer.add_scalar('train/_loss', train_metric['epoch_train_loss'], epoch)
            writer.add_scalar('val/_loss', val_metric['epoch_test_loss'], epoch)
            writer.add_scalar('train/_mae', train_metric['epoch_train_mae'], epoch)
            writer.add_scalar('val/_mae', val_metric['epoch_test_mae'], epoch)
            writer.add_scalar('train/_mse', train_metric['epoch_train_mse'], epoch)
            writer.add_scalar('val/_mse', val_metric['epoch_test_mse'], epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

            if mode == 'classification':
                writer.add_scalar('train/_accuracy', train_metric['epoch_train_accuracy'], epoch)
                writer.add_scalar('val/_accuracy', val_metric['epoch_test_accuracy'], epoch)

            if epoch > 3 and min_val_mae > val_metric['epoch_test_mae']:
                print("Saving model ....")
                min_val_mae = val_metric['epoch_test_mae']
                model_content = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                                 'val_mae': val_metric['epoch_test_mae'], 'train_mae': train_metric['epoch_train_mae']}
                torch.save(model_content, kwargs['model_dir'] / f"model_epoch_{epoch}.pt")
    writer.close()


def init(params: dict) -> dict:
    if params['device'] in ['auto', 'gpu']:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("GPU not available, using cpu only.")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    params['device'] = device

    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
    exp_dir = Path(params['exp_dir'])
    model_dir = exp_dir / 'model'
    if not model_dir.exists():
        model_dir.mkdir(parents=True)
    else:
        print("Model dir already exists")
        # sleep(10)
    params['exp_dir'] = exp_dir
    params['model_dir'] = model_dir
    return params


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

    train_config = config['Train']
    train_config['device'] = setup['device']
    train_config['model_dir'] = setup['model_dir']
    train_config['exp_dir'] = setup['exp_dir']

    dataset_config['mode'] = mode
    dataset_config['pos_enc_dim'] = model_config['pos_enc_dim']

    dataset_map = {'Mohler': MohlerDataset, 'Sbank': SbankDataset, 'FakeNews': FakeNews}
    dataset = dataset_map[dataset_name](**dataset_config)
    # if lap_pos_enc:
    #     dataset.add_laplacian_positional_encodings(model_config['pos_enc_dim'])
    # print("Getting Spacy embeddings .. ")
    # word_embeddings = get_embedding_weights()
    print("Done.")
    # word_embeddings = torch.from_numpy(word_embeddings.astype(np.float32))
    gt_model = GraphTransformerNet(**model_config)
    train_model(gt_model, dataset, mode=mode, **train_config)
