try:
    from tqdm import tqdm, trange
except ImportError:
    def tqdm(iterable):
        return iterable


    def trange(iterable):
        return iterable
import sys
import os

sys.path.append(os.getcwd())
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader, TwitterTennisDatasetLoader, \
    PedalMeDatasetLoader, WikiMathsDatasetLoader  # , \
from torch_geometric_temporal.signal import temporal_signal_split

printFiles = True
import argparse
from src.temporal_tde_gnn import tdegnn_temporal

parser = argparse.ArgumentParser(description="Temporal Benchmark")
parser.add_argument(
    "--user", default='l', type=str, help="user name")
parser.add_argument(
    "--dataset",
    default='wiki',
    type=str,
    help='dataset name',
)
parser.add_argument(
    "--outputDir",
    default='temporal_outputs',
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
    "--cumulative",
    default=0,
    type=int,
    help='if 0 learn incremental otherwise cumulative',
)

parser.add_argument(
    "--order",
    default=1,
    type=int,
    help='order',
)

parser.add_argument(
    "--useMHA",
    default=0,
    type=int,
    help='if 1 use MHA otherewise direct parameterization',
)

parser.add_argument(
    "--sharedWeights",
    default=0,
    type=int,
    help='if 0 share weights, if 1 reuse them',
)

parser.add_argument(
    "--channels",
    default=64,
    type=int,
    help='num of channels',
)

parser.add_argument(
    "--timeEmbedding",
    default=1,
    type=int,
    help='num of channels',
)

parser.add_argument(
    "--layers",
    default=8,
    type=int,
    help='num of channels',
)

parser.add_argument(
    "--addU0",
    default=0,
    type=int,
    help='if to save to log file or print',
)

parser.add_argument(
    "--multLayers",
    default=0,
    type=int,
    help='if to save to log file or print',
)

parser.add_argument(
    "--explicit",
    default=1,
    type=int,
    help='if to save to log file or print',
)

parser.add_argument(
    "--pred",
    default=1,
    type=int,
    help='if to save to log file or print',
)


def mae(pred, gt):
    mask = (gt != 0).float()
    mask /= torch.mean(mask)
    mae = torch.abs(pred - gt)
    mae = mae * mask
    mae[mae != mae] = 0

    return mae.mean()


def rmse(pred, gt):
    mask = (gt != 0).float()
    mask /= torch.mean(mask)
    rmse = torch.sqrt(F.mse_loss(pred, gt))
    rmse[rmse != rmse] = 0

    rmse = rmse * mask
    rmse[rmse != rmse] = 0
    return rmse.mean()


def mape(pred, gt):
    mask = (gt != 0).float()
    mask /= torch.mean(mask)
    mape = torch.abs((pred - gt)) / torch.abs(gt)
    mape = mape * mask
    mape[mape != mape] = 0
    return mape.mean()


args = parser.parse_args()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
nsplits = 10
num_epochs = 100
datastr = args.dataset

base_path = '.'

outputpath = os.path.join(base_path, args.outputDir)
datapath = os.path.join(base_path, 'temporal_data')
if not os.path.exists(outputpath):
    os.mkdir(outputpath)
if not os.path.exists(datapath):
    os.mkdir(datapath)
datapath = os.path.join(datapath, args.dataset)

split_test_costs = []
num_features = 4
num_output = 1
lrReact = 1e-3
lrDiffusion = 1e-3
lrProj = 1e-3
lrMHA = 1e-3
lrmHA_factor = 1e-3
wdMHA = 1e-5
wdMHA_factor = 0
wdReact = 1e-3
wdDiffusion = 0
wdProj = 1e-4
dropout = 0.0
dropoutOC = 0.0
dt = 0.1
mha_dropout = 0.1
n_channels = 64

for splitIdx in trange(nsplits, desc='nsplits'):
    torch.manual_seed(splitIdx)
    np.random.seed(splitIdx)
    if datastr.lower() == 'pedalme':
        loader = PedalMeDatasetLoader()
        num_features = 4
    elif datastr.lower() == 'cpox':
        loader = ChickenpoxDatasetLoader()
        num_features = 4
    elif datastr.lower() == 'wiki':
        loader = WikiMathsDatasetLoader()
        num_features = 8
    else:
        print('no such dataset, exit.')
        exit()
    dataset = loader.get_dataset()
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.9)
    model = tdegnn_temporal(nlayers=args.layers, nhid=n_channels, nin=num_features, nout=num_output, dropout=dropout,
                   h=dt, sharedWeights=args.sharedWeights, addU0=args.addU0,
                   multiplicativeReaction=args.multLayers, explicit=args.explicit, dropoutOC=dropoutOC,
                   metrpems=False, mha_dropout=mha_dropout, useMHA=args.useMHA, baseline=args.baseline,
                   order=args.order)
    model.reset_parameters()
    model = model.to(device)

    optimizer = torch.optim.Adam([
        {'params': model.reactionParams.parameters(), 'lr': lrReact, 'weight_decay': wdReact},
        {'params': model.diffusionParams.parameters(), 'lr': lrDiffusion, 'weight_decay': wdDiffusion},
        {'params': model.projParams.parameters(), 'lr': lrProj, 'weight_decay': wdProj},
        {'params': model.mha.parameters(), 'lr': lrMHA, 'weight_decay': wdMHA},
        {'params': model.mha_factor, 'lr': lrmHA_factor, 'weight_decay': wdMHA_factor},
        {'params': model.C.parameters(), 'lr': lrMHA, 'weight_decay': 0},

    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    max_patience = 20
    bad_counter = 0


    def createTE(time, nfreqs=10, lags=4):
        freqs = np.arange(1, nfreqs + 1).reshape(1, 1, nfreqs)
        time = np.arange(time * lags, time * (lags) + lags)
        time_step = np.expand_dims(time, -1) * freqs
        cosTime = torch.from_numpy(np.cos(2 * np.pi * time_step)).permute(0, 2, 1)
        sinTime = torch.from_numpy(np.sin(2 * np.pi * time_step)).permute(0, 2, 1)
        time_feature = torch.cat([cosTime, sinTime], dim=1)
        time_feature = time_feature.repeat(snapshot.x.shape[0], 1, 1).to(device)
        return time_feature.float()


    def eval_test():
        model.eval()
        cost = 0
        test_rmse = 0
        test_mape = 0
        with torch.no_grad():
            for time, snapshot in enumerate(tqdm(test_dataset, desc='Testing')):
                snapshot = snapshot.to(device)
                snapshot = snapshot.to(device)
                actual_time = time + len(train_dataset.targets)
                time_feature = createTE(actual_time, nfreqs=10, lags=model.nin)
                y_hat = model(snapshot.x, time_feature, snapshot.edge_index, regression=True,
                              edge_attr=snapshot.edge_attr)
                y_true = snapshot.y
                loss = ((y_hat.squeeze() - y_true) ** 2).mean()
                cost += loss.mean().item()
            cost = cost / (time + 1)

        model.train()
        return cost, test_rmse, test_mape


    best_test_cost = 9999999
    best_train_cost = 999999

    best_test_rmse = 999999
    best_test_mape = 10000

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        cost = 0

        if args.cumulative:
            for time, snapshot in enumerate(tqdm(train_dataset, desc='Training')):
                snapshot = snapshot.to(device)
                tt = time if args.timeEmbedding else None
                y_hat = model(snapshot.x, snapshot.edge_index, regression=True)
                cost = cost + torch.mean((y_hat - snapshot.y).abs())

            cost = cost / (time + 1)
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            train_cost = 0
            train_rmse = 0
            train_mape = 0
            for time, snapshot in enumerate(tqdm(train_dataset, desc='Training')):
                snapshot = snapshot.to(device)
                time_feature = createTE(time=time, nfreqs=10, lags=model.nin)
                y_hat = model(snapshot.x, time_feature, snapshot.edge_index, regression=True,
                              edge_attr=snapshot.edge_attr)
                y_true = snapshot.y
                loss = ((y_hat.squeeze() - y_true) ** 2).mean()
                cost = loss  # .mean()
                cost.backward()
                optimizer.step()
                train_cost += cost.item()
                optimizer.zero_grad()

        train_cost = cost
        test_cost, test_rmse, test_mape = eval_test()
        bad_counter += 1
        if test_cost < best_test_cost and test_cost == test_cost:
            bad_counter = 0
            best_test_cost = test_cost
            best_test_rmse = test_rmse
            best_test_mape = test_mape
            best_train_cost = train_cost
        if bad_counter >= max_patience:
            scheduler.step()

    print("Split:", splitIdx, ", best test cost:", best_test_cost,
          ", best train cost:", best_train_cost, flush=True)
    split_test_costs.append(best_test_cost)
print("All test costs:", split_test_costs, ", mean:", np.array(split_test_costs).mean(), ", std:",
      np.array(split_test_costs).std(), flush=True)
