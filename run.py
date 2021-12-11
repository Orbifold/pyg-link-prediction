#########################################################
# Code for the article ''
# https://bratanic-tomaz.medium.com
# MIT License
#
#########################################################

import numpy as np
from datetime import datetime

from pokec import Pokec
import torch_geometric.transforms as T
import torch, signal, csv, os
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from pokec import Pokec

# Load the Pyg dataset
p = Pokec()
p.describe()
data = p.data

# pyg transformation of the data splitting in test, val, train sets
transform = T.Compose([
    T.ToUndirected(merge = True),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val = 0.0005, num_test = 0.0001, is_undirected = True, add_negative_train_samples = False),
])
train_data, val_data, test_data = transform(data)

# the larger the batch size the faster things will be
batch_size = 2048

# define batch loaders for the three sets
train_loader = NeighborLoader(data, num_neighbors = [10] * 2, shuffle = True, input_nodes = data.train_mask, batch_size = batch_size)
val_loader = NeighborLoader(data, num_neighbors = [10] * 2, input_nodes = data.val_mask, batch_size = batch_size)
test_loader = NeighborLoader(data, num_neighbors = [10] * 2, input_nodes = data.test_mask, batch_size = batch_size)

# the actual Pyg network
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        # chaining two convolutions with a standard relu activation

        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        # cosine similarity
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim = -1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple = False).t()


model = Net(data.num_features, 128, 64).to(device)
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.01)
# BCELoss creates a criterion that measures the Binary Cross Entropy between the target and the output.
criterion = torch.nn.BCEWithLogitsLoss()


def train():
    """
    Single epoch model training in batches.
    :return: total loss for the epoch
    """
    model.train()
    total_examples = total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_size = batch.batch_size
        z = model.encode(batch.x, batch.edge_index)
        neg_edge_index = negative_sampling(edge_index = batch.edge_index, num_nodes = batch.num_nodes, num_neg_samples = None, method = 'sparse')
        edge_label_index = torch.cat([batch.edge_index, neg_edge_index], dim = -1, )
        edge_label = torch.cat([torch.ones(batch.edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim = 0)
        out = model.decode(z, edge_label_index).view(-1)
        # loss = criterion(out[:batch_size], edge_label[:batch_size])
        loss = criterion(out, edge_label)
        # standard torch mechanics here
        loss.backward()
        optimizer.step()
        total_examples += batch_size
        total_loss += float(loss) * batch_size
    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    """
        Evalutes the model on the test set.

    :param loader: the batch loader
    :return: a score
    """
    model.eval()
    scores = []
    threshold = torch.tensor([0.7])
    for batch in tqdm(loader):
        z = model.encode(batch.x, batch.edge_index)
        out = model.decode(z, batch.edge_index).view(-1).sigmoid()
        pred = (out > threshold).float() * 1
        score = f1_score(np.ones(batch.edge_index.size(1)), pred.cpu().numpy())
        scores.append(score)
    return np.average(scores)


def handler(signum, frame):
    """
        Keyboard interrupt handler.
        Use with

            signal.signal(signal.SIGINT, handler)

    :param signum:
    :param frame:
    """
    res = input("Do you really want to stop the training loop? (y/n)")
    if res == 'y':
        exit(1)


def load_model(run_id):
    """
        Returns a saved model.

    :param run_id: the model id to load
    :return: a hydrated model
    """
    if not os.path.exists(f"model_{run_id}"):
        raise Exception(f"Model id '{run_id}' does not exist.")
    model = Net(data.num_features, 128, 64).to(device)
    model.load_state_dict(torch.load(f"model_{run_id}"))
    model.eval()
    return model


def predictions(run_id, max = 1000, threshold = 0.99):
    """
        Creates predictions for the specified run.

    :param run_id: model id
    :param max: the maximum amount of predictions to output
    """
    pred_edges = []
    model = load_model(run_id)

    loader = NeighborLoader(data, num_neighbors = [10] * 2, shuffle = True, input_nodes = None, batch_size = batch_size)
    threshold_tensor = torch.tensor([threshold])
    for batch in tqdm(loader):
        z = model.encode(batch.x, batch.edge_index)
        # collecting negative edge tuples ensure that the decode are actual non-existing edges
        neg_edge_index = negative_sampling(edge_index = batch.edge_index, num_nodes = None, num_neg_samples = None, method = 'sparse')
        out = model.decode(z, neg_edge_index).view(-1).sigmoid()
        pred = ((out > threshold_tensor).float() * 1).cpu().numpy()
        found = np.argwhere(pred == 1)
        if found.size > 0:
            edge_tuples = neg_edge_index.t().cpu().numpy()
            select_index = found.reshape(1, found.size)[0]
            edges = edge_tuples[select_index]
            pred_edges += edges.tolist()
            if len(pred_edges) >= max:
                break

    with open(f"predictions_{run_id}.csv", "wt") as f:
        w = csv.writer(f)
        w.writerow(["source", "target"])
        for s, t in pred_edges:
            w.writerow([s, t])


def run():
    """
        Run the training and makes predictions.

    """
    run_id = int(datetime.timestamp(datetime.now()))
    writer = SummaryWriter(f"runs/{run_id}")

    start_time = datetime.now()
    epochs = 5
    with trange(epochs + 1) as t:
        for epoch in t:
            try:
                t.set_description('Epoch %i/%i train' % (epoch, epochs))
                loss = train()
                t.set_description('Epoch %i/%i test' % (epoch, epochs))
                val_acc = test(test_loader)
                t.set_postfix(loss = loss, accuracy = val_acc)
                writer.add_scalar('loss', loss, epoch)
                writer.add_scalar('accuracy', val_acc, epoch)
                print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {val_acc:.4f}")
            except KeyboardInterrupt:
                break
    writer.close()
    torch.save(model.state_dict(), f"model_{run_id}")
    time_elapsed = datetime.now() - start_time
    print("Creating predictions")
    predictions(run_id)
    print(f"\nRun {run_id}:")
    print(f"\tEpochs: {epoch}")
    print(f"\tTime: {time_elapsed}")
    print(f"\tAccuracy: {val_acc * 100:.01f}")
    print(f"\tParameters saved to 'model_{run_id}'.")
    print(f"\tPredictions saved to 'predictions_{run_id}.csv'.")


run()
# if you want to visualize things
# install tensorboard (conda install tensorboard)
# and run:
#   tensorboard --logdir=runs
