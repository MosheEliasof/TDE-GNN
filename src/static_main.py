import sys
import os

sys.path.append(os.getcwd())
import torch
import torch.nn.functional as F
import process
import utils
from torch_geometric.utils import sparse as sparseConvert
import argparse
from src.static_tdegnn import static_tdegnn

parser = argparse.ArgumentParser(description="Node Classification")
parser.add_argument(
    "--user", default='l', type=str, help="user name")
parser.add_argument(
    "--dataset",
    default='cora',
    type=str,
    help='dataset name',
)

parser.add_argument(
    "--order",
    default='1',
    type=int,
    help='order',
)
parser.add_argument(
    "--outputDir",
    default='tdgnn_node_output',
    type=str,
    help='dataset name',
)

parser.add_argument(
    "--baseline",
    default=0,
    type=int,
    help='if to save to log file or print',
)

parser.add_argument(
    "--saveInLog",
    default=0,
    type=int,
    help='if to save to log file or print',
)

parser.add_argument(
    "--sharedWeights",
    default=0,
    type=int,
    help='if to share weights between layers',
)

parser.add_argument(
    "--nlayers",
    default=2,
    type=int,
    help='if to share weights between layers',
)

parser.add_argument(
    "--useMHA",
    default=0,
    type=int,
    help='if use MHA or direct parameterization (if 0)',
)

parser.add_argument(
    "--explicit",
    default=1,
    type=int,
    help='if to use explicit time discretization',
)
parser.add_argument(
    "--addU0",
    default=0,
    type=int,
    help='if add U0',
)
parser.add_argument(
    "--useBN",
    default=1,
    type=int,
    help='if use batchnorm',
)
parser.add_argument(
    "--multLayers",
    default=0,
    type=int,
    help='if to use multiplicative reaction layer',
)
parser.add_argument(
    "--useReaction",
    default=1,
    type=int,
    help='if to use a reaction term',
)
parser.add_argument(
    "--useDiffusion",
    default=1,
    type=int,
    help='if to use a diffusion term',
)
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
args = parser.parse_args()
datastr = args.dataset
if datastr == "cora":
    num_output = 7
elif datastr == "citeseer":
    num_output = 6
elif datastr == "pubmed":
    num_output = 3
elif datastr == "chameleon":
    num_output = 5
else:
    num_output = 5
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
args = parser.parse_args()
printFiles = True
datastr = args.dataset
base_path = './'
printFiles = False

outputpath = os.path.join(base_path, args.outputDir)
if not os.path.exists(outputpath):
    os.mkdir(outputpath)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
lrReact = 1e-3
lrDiffusion = 1e-4
lrProj = 1e-3
lrMHA = 1e-5
lrScale = 1e-5
wdReact = 1e-4
wdDiffusion = 1e-5
wdProj = 1e-4
wdMHA = 1e-5
dropout = 0.5
dropoutOC = 0.5
n_channels = 64
dt = 0.1
lpDiffusion = 1
useBN = 1
mha_dropout = 0.2


def train_step(model, optimizer, features, labels, adj, idx_train):
    model.train()
    optimizer.zero_grad()
    xn = features.squeeze().t()
    pred = model(xn, adj, regression=False)
    acc_train = utils.accuracy(pred[idx_train], labels[idx_train].to(device))
    lossNLL = F.nll_loss(pred[idx_train, :], labels[idx_train])
    loss = lossNLL
    loss.backward()
    optimizer.step()
    return loss.item(), acc_train.item()


def eval_test_step(model, features, labels, adj, idx_test):
    model.eval()
    with torch.no_grad():
        xn = features.squeeze().t()
        pred = model(xn, adj, regression=False)
        loss_test = F.nll_loss(pred[idx_test], labels[idx_test].to(device))
        acc_test = utils.accuracy(pred[idx_test], labels[idx_test].to(device))
        return loss_test.item(), acc_test.item()


def train(datastr, splitstr, num_output):
    slurm = (args.user == 'b') or (args.user == 's') or (args.user == 'e') or (
            args.saveInLog == 1)
    adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = process.full_load_data(
        datastr,
        splitstr, slurm=slurm)
    adj = adj.to_dense()
    [edge_index, edge_weight] = sparseConvert.dense_to_sparse(adj)
    del adj
    edge_index = edge_index.to(device)
    features = features.to(device).t().unsqueeze(0)
    idx_train = idx_train.to(device)
    idx_test = idx_test.to(device)
    labels = labels.to(device)

    model = static_tdegnn(nlayers=args.nlayers, nin=num_features, nhid=n_channels, nout=num_output,
                   dropout=dropout, h=dt, sharedWeights=args.sharedWeights, addU0=args.addU0,
                   multiplicativeReaction=args.multLayers, dropoutOC=dropoutOC, explicit=args.explicit,
                   useBN=useBN,
                   useDiffusion=args.useDiffusion, useReaction=args.useReaction,
                   lpDiffusion=lpDiffusion, mha_dropout=mha_dropout,
                   useMHA=args.useMHA, baseline=args.baseline, order=args.order)
    model.reset_parameters()
    model = model.to(device)
    optimizer = torch.optim.Adam([
        {'params': model.reactionParams.parameters(), 'lr': lrReact, 'weight_decay': wdReact},
        {'params': model.diffusionParams.parameters(), 'lr': lrDiffusion, 'weight_decay': wdDiffusion},
        {'params': model.projParams.parameters(), 'lr': lrProj, 'weight_decay': wdProj},
        {'params': model.mha.parameters(), 'lr': lrMHA, 'weight_decay': wdMHA},
        {'params': model.mha_factor, 'lr': lrScale, 'weight_decay': 0},
        {'params': model.C.parameters(), 'lr': lrMHA, 'weight_decay': 0},
    ])
    best_test = 0
    best_val = 0
    max_patience = 201
    patience = 0
    for epoch in range(2001):
        loss_tra, acc_tra = train_step(model, optimizer, features, labels, edge_index, idx_train)
        loss_test, acc_test = eval_test_step(model, features, labels, edge_index, idx_test)
        loss_val, acc_val = eval_test_step(model, features, labels, edge_index, idx_val)
        if acc_val > best_val:
            best_val = acc_val
            best_test = acc_test
            patience = 0
        print(acc_test)
        if patience > max_patience:
            break
        patience = patience + 1
    return best_test


accs = []
for i in range(10):
    # splits are the official 10 splits from Geom-GCN (Pei et al., can be found on PyTorch-Geometric and also here: https://github.com/graphdml-uiuc-jlu/geom-gcn)
    splitstr = '../splits/' + datastr + '_split_0.6_0.2_' + str(i) + '.npz'
    if (args.user == "b") or (args.user == "s") or (args.user == "e") or (args.saveInLog == 1):
        splitstr = 'splits/' + datastr + '_split_0.6_0.2_' + str(i) + '.npz'
    cbest = train(datastr, splitstr, num_output)
    print("Split #", i, ", best:", cbest)

    accs.append(cbest)
print("accs:", accs, flush=True)
print("mean:", np.array(accs).mean(), flush=True)
print("std:", np.array(accs).std(), flush=True)
