import torch
from metrics import MAE, MSE


def evaluate_network(model, device, data_loader):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    epoch_test_mse = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['tokens'].to(device).to(torch.long)  # num x feat
            batch_e = batch_graphs.edata['type'].to(device).to(torch.long)
            batch_targets = batch_targets.to(device)
            try:
                batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            except:
                batch_lap_pos_enc = None

            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc)
            loss = model.loss(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(batch_scores, batch_targets)
            epoch_test_mse += MSE(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)

        epoch_test_mse /= (iter + 1)
        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)

    return epoch_test_loss, epoch_test_mae, epoch_test_mse
