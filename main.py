# import the pytorch library into environment and check its version
import os
import torch
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, GraphConv

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.utils import stochastic_blockmodel_graph
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
import networkx as nx
import pickle
from scipy.sparse import csr_matrix, save_npz, load_npz
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.loader import NeighborSampler
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



print("Using torch", torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device is {device}")



adj_matrix = load_npz("data/adj_matrix.npz")
with open('data/cluster_id.pkl', 'rb') as file:
      cluster_ids = pickle.load(file)
cluster_ids = torch.Tensor(cluster_ids)

feature_size = 512
num_classes = len(torch.unique(cluster_ids))
num_nodes = adj_matrix.shape[0]


# x = torch.randn((num_nodes, feature_size))  # random feature
x = torch.Tensor(np.load(f"data/embedding_{feature_size}.npy")) # augmented feature
y = cluster_ids  

edge_index, _ = from_scipy_sparse_matrix(adj_matrix)

train_size = int(num_nodes * 0.7)
val_size = int(num_nodes * 0.15)
test_size = num_nodes - train_size - val_size

indices = np.arange(num_nodes)
np.random.shuffle(indices)
train_idx = torch.tensor(indices[:train_size], dtype=torch.long)
val_idx = torch.tensor(indices[train_size:train_size + val_size], dtype=torch.long)
test_idx = torch.tensor(indices[train_size + val_size:], dtype=torch.long)

BATCH_SIZE = 256
train_loader = NeighborSampler(edge_index, node_idx=train_idx, sizes=[10, 10], batch_size=BATCH_SIZE, shuffle=True)
val_loader = NeighborSampler(edge_index, node_idx=val_idx, sizes=[10, 10], batch_size=BATCH_SIZE, shuffle=False)
test_loader = NeighborSampler(edge_index, node_idx=test_idx, sizes=[10, 10], batch_size=BATCH_SIZE, shuffle=False)


# Step 2: Define a 2-layer GCN

class GCN(nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList([
            GraphConv(num_features, hidden_size),
            GraphConv(hidden_size, hidden_size)
        ])
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.prediction_head = nn.Linear(hidden_size, num_classes)

    def forward(self, x, adjs, return_embedding=False):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if return_embedding and i == len(adjs) - 1:
                return x 
            if i != len(adjs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        x = self.prediction_head(self.linear(x))  
        return x

class GAT(nn.Module):
    def __init__(self, num_features, hidden_size, num_classes, heads=8, dropout=0.5):
        super(GAT, self).__init__()
        self.convs = nn.ModuleList([
            GATConv(num_features, hidden_size, heads=heads, dropout=dropout),
            GATConv(hidden_size * heads, hidden_size, heads=1,concat=False, dropout=dropout)
        ])
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.prediction_head = nn.Linear(hidden_size, num_classes)

    def forward(self, x, adjs, return_embedding=False):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)

            if i != len(adjs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        if return_embedding:
                return x

        x = F.relu(self.linear(x))
        x = self.prediction_head(x) 
        return x
    


hidden_size = 512
model = GCN(feature_size, hidden_size, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Step 4: Train the GNN

def train(loader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_size, n_id, adjs in loader:
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        out = model(x[n_id].to(device), adjs)
        labels = y[n_id[:batch_size]].to(device).long()
        loss = F.cross_entropy(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(out.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    accuracy = correct / total
    return total_loss / len(loader), accuracy

def validate(loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_size, n_id, adjs in loader:
            adjs = [adj.to(device) for adj in adjs]
            out = model(x[n_id].to(device), adjs)
            labels = y[n_id[:batch_size]].to(device).long()
            loss = F.cross_entropy(out, labels)
            total_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return total_loss / len(loader), accuracy


# Training and validation process
train_losses, val_losses = [], []
num_epochs = 100  # Set the number of epochs

for epoch in range(num_epochs):
    train_loss, train_acc = train(train_loader)
    val_loss, val_acc = validate(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}')

# Plotting the training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
def extract_embeddings(loader):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch_size, n_id, adjs in loader:
            # adjs = [(edge_index.to(device), size) for edge_index, size in adjs]
            adjs = [(adjs[0].to(device), adjs[1], adjs[-1])]
            out = model(x[n_id].to(device), adjs, return_embedding=True)
            embeddings.append(out.cpu())
    return torch.cat(embeddings, dim=0)

full_loader = NeighborSampler(edge_index, node_idx=None, sizes=[-1], batch_size=1024, shuffle=False)
embeddings = extract_embeddings(full_loader)

def get_predictions(loader):
    model.eval()
    predictions = []
    logits_list = []
    with torch.no_grad():
        for batch_size, n_id, adjs in loader:
            adjs = [(adjs[0].to(device), adjs[1], adjs[-1])]
            logits = model(x[n_id].to(device), adjs, return_embedding=False)

            predicted_classes = logits.argmax(dim=-1)
            predictions.append(predicted_classes.cpu())
            logits_list.append(logits.cpu())
    return torch.cat(predictions, dim=0), torch.cat(logits_list, dim=0)

# Obtain predictions
full_loader = NeighborSampler(edge_index, node_idx=None, sizes=[-1], batch_size=1024, shuffle=False)
predictions, logits = get_predictions(full_loader)



def evaluate_classification_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    metrics = {}
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['Recall'] = recall_score(y_true, y_pred, average='weighted')
    metrics['F1 Score'] = f1_score(y_true, y_pred, average='weighted')


    return metrics

metrics = evaluate_classification_metrics(y, predictions)
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value}")


pca = PCA(n_components=2)
reduced_data = pca.fit_transform(embeddings.numpy())
embeddings_2d = reduced_data
# tsne = TSNE(n_components=2)
# embeddings_2d = tsne.fit_transform(reduced_data)

plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=y.numpy(), cmap='jet')
plt.colorbar()
plt.title('Node Embeddings Visualization using PCA')
plt.show()