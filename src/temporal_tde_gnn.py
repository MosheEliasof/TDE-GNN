import torch
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros
import torch.nn as nn
from torch_geometric.utils import get_laplacian
from torch_geometric.utils import add_self_loops, remove_self_loops



class tdegnn_temporal(nn.Module):
    def __init__(self, nlayers, nhid, nin, nout, dropout=0.5, dropoutOC=0.5, h=0.1, timeEmbeddingFreqs=10,
                 sharedWeights=False,
                 multiplicativeReaction=True,
                 addU0=True, explicit=False, metrpems=True, mha_dropout=0, useMHA=True, order=None, baseline=False):
        super(tdegnn_temporal, self).__init__()
        self.baseline = baseline
        self.order = order if order is not None else min(nlayers, nin)
        self.useMHA = useMHA
        self.mha_dropout = mha_dropout
        self.metrpems = metrpems
        self.dropoutOC = dropoutOC
        self.explicit = explicit
        self.multiplicativeReaction = multiplicativeReaction
        self.addU0 = addU0
        self.sharedWeights = sharedWeights
        self.timeEmbeddingFreqs = timeEmbeddingFreqs
        self.dropout = dropout
        self.h = h
        self.nin = nin
        self.nhid = nhid
        self.nout = nout
        self.nlayers = nlayers
        # Open parameters:
        self.KopenHist = torch.nn.Linear(nin, nhid)
        self.KopenState = torch.nn.Linear(1, nhid)
        self.KopenHistBN = torch.nn.BatchNorm1d(nhid)
        self.KopenStateBN = torch.nn.BatchNorm1d(nhid)
        self.HistEmbed_conv1d = torch.nn.Conv1d(in_channels=1, out_channels=nhid, kernel_size=1)

        # Reaction parameters
        self.KR1 = torch.nn.ModuleList()
        self.KR2 = torch.nn.ModuleList()
        self.reactionBNs = torch.nn.ModuleList()

        if self.addU0:
            self.KRU0_1 = torch.nn.ModuleList()

        for i in range(self.nlayers):
            self.reactionBNs.append(torch.nn.BatchNorm1d(nhid))
            self.KR1.append(torch.nn.Linear(nhid, 1 * nhid))
            self.KR2.append(torch.nn.Linear(1 * nhid, nhid))
            if self.addU0:
                self.KRU0_1.append(torch.nn.Linear(nhid, nhid))

        # Diffusion coefficients:
        self.Kappa = torch.nn.ParameterList()

        for i in range(self.nlayers):
            self.Kappa.append(torch.nn.Parameter(1e-3 * torch.torch.randn(nhid)))

        # Hist embedding:
        self.HistEmbeds = torch.nn.ModuleList()
        self.HistBNs = torch.nn.ModuleList()
        for i in range(self.nlayers):
            self.HistEmbeds.append(torch.nn.Linear(3 * nhid, nhid))
            self.HistBNs.append(torch.nn.BatchNorm1d(nhid))
        # Time Embedding layers:
        self.initTimeEmbed = torch.nn.Conv1d(in_channels=2 * self.timeEmbeddingFreqs, out_channels=1, kernel_size=1,
                                             padding=0)
        self.rescales = torch.nn.ModuleList()
        self.rescalesBNs = torch.nn.ModuleList()

        for i in range(self.nlayers):
            self.rescales.append(torch.nn.Linear((self.nin) + self.nhid, self.nhid))
            self.rescalesBNs.append(torch.nn.BatchNorm1d(nhid))
        # Closing parameters:
        self.Kclose = torch.nn.Linear(nhid, nout)

        self.edge_scale = torch.nn.Linear(1, self.nhid)

        # Create optimization groups:
        # Create reaction groups:
        self.reactionParams = torch.nn.ModuleList()
        self.reactionParams.append(self.reactionBNs)
        self.reactionParams.append(self.KR1)
        self.reactionParams.append(self.KR2)
        if self.addU0:
            self.reactionParams.append(self.KRU0_1)

        # Create diffusion groups:
        self.diffusionParams = torch.nn.ModuleList()
        self.diffusionParams.append(self.Kappa)

        # Create open close group:
        self.projParams = torch.nn.ModuleList()
        self.projParams.append(self.Kclose)
        self.projParams.append(self.rescales)
        self.projParams.append(self.rescalesBNs)
        self.projParams.append(self.edge_scale)
        self.projParams.append(self.KopenState)
        self.projParams.append(self.KopenHist)
        self.projParams.append(self.KopenStateBN)
        self.projParams.append(self.KopenHistBN)
        self.projParams.append(self.initTimeEmbed)
        self.projParams.append(self.HistEmbed_conv1d)

        self.mha = torch.nn.MultiheadAttention(embed_dim=nhid, num_heads=1, dropout=self.mha_dropout)
        self.mha_factor = torch.nn.Parameter(torch.Tensor([10]))

        self.C = torch.nn.ParameterList()
        for jj in range(nlayers):
            tmp = torch.zeros(jj + 1)
            tmp[-1] = 1
            self.C.append(torch.nn.Parameter(tmp))

    def reset_parameters(self):
        for i in range(self.nlayers):
            self.KR1[i].weight = torch.nn.Parameter(
                torch.eye(n=self.KR1[i].weight.shape[0], m=self.KR1[i].weight.shape[1]) + 1e-2 * torch.randn(
                    self.KR1[i].weight.shape))
            self.KR2[i].weight = torch.nn.Parameter(
                torch.eye(self.KR2[i].weight.shape[0], self.KR2[i].weight.shape[1]) + + 1e-2 * torch.randn(
                    self.KR2[i].weight.shape))
            self.KR1[i].bias = torch.nn.Parameter(torch.zeros_like(self.KR1[i].bias))
            self.KR2[i].bias = torch.nn.Parameter(torch.zeros_like(self.KR2[i].bias))

            if self.addU0:
                self.KRU0_1[i].weight = torch.nn.Parameter(
                    torch.eye(self.KRU0_1[i].weight.shape[0], self.KRU0_1[i].weight.shape[1]) + 1e-2 * torch.randn(
                        self.KRU0_1[i].weight.shape))
                self.KRU0_1[i].bias = torch.nn.Parameter(torch.zeros_like(self.KRU0_1[i].bias))

        glorot(self.KopenHist.weight)
        glorot(self.KopenState.weight)

        glorot(self.Kclose.weight)

        zeros(self.KopenHist.bias)
        zeros(self.KopenState.bias)

        zeros(self.Kclose.bias)

    def reaction(self, Tstate, Thist, T0, layer):
        dT = self.KR1[layer](Thist)
        if self.addU0:
            dT += self.KRU0_1[layer](T0)
        if self.multiplicativeReaction:
            dT2 = self.KR2[layer](Thist)
            dT = dT + (Tstate * F.hardtanh(dT2))

        Tstate = F.relu(self.reactionBNs[layer](Tstate + self.h * dT))

        return Tstate

    def diffusion(self, Tstate, Thist, edge_index, layer, explicit=False, edge_attr=None):
        cgiter = 5
        Kd = self.Kappa[layer]  # (Tstate)
        Kd = F.hardtanh(Kd, max_val=1, min_val=0)
        if explicit:
            lap_edge_index, lap_edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=Tstate.shape[0],
                                                            edge_weight=edge_attr)
            lap_op = torch.sparse_coo_tensor(indices=lap_edge_index, values=lap_edge_weight,
                                             size=(Tstate.shape[0], Tstate.shape[0]))

            LapY = torch.sparse.mm(lap_op, Tstate)
            return Tstate - self.h * LapY * Kd

        def matvec(Y, edge_index, h=self.h):
            lap_edge_index, lap_edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=Y.shape[0],
                                                            edge_weight=edge_attr)
            lap_op = torch.sparse_coo_tensor(indices=lap_edge_index, values=lap_edge_weight,
                                             size=(Y.shape[0], Y.shape[0]))

            LapY = torch.sparse.mm(lap_op, Y)
            # Diffusion coeff
            LapY = Kd * LapY
            out = Y + h * (LapY)
            return out

        X = Tstate.clone()
        R = Tstate.clone() - matvec(X, edge_index)
        P = R.clone()
        normRold = (R * R).mean()
        for i in range(cgiter):
            LP = matvec(P, edge_index)
            alpha = (R * R).mean() / (torch.mean(P * LP) + 1e-6)
            X = X + alpha * P
            R = R - alpha * LP
            normR = (R * R).mean()
            if normR < 1e-5:
                break
            beta = normR / (normRold + 1e-6)
            normRold = normR.clone()
            P = R + beta * P

        return X

    def forward(self, T, time_feature, edge_index, regression=False, edge_attr=None):
        if not self.metrpems:
            T = F.dropout(T, p=self.dropout, training=self.training)
        else:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=T.shape[0])

        Thist = T.clone()
        Thist = F.relu(self.KopenHistBN(self.KopenHist(Thist)))  # n x 5
        Thist_embed = T.clone().unsqueeze(1)
        Thist_embed = F.relu(self.KopenStateBN(self.HistEmbed_conv1d(Thist_embed)))
        Thist_embed = Thist_embed.permute(2, 0, 1)
        acts = Thist_embed.clone()[:self.order, :, :]
        Tstate = Thist_embed[-1, :, :]
        T0Hist = Thist.clone()


        timeEmbedding0 = F.silu(self.initTimeEmbed(time_feature).squeeze())
        if timeEmbedding0.dim() == 1:
            timeEmbedding0 = timeEmbedding0.unsqueeze(-1)
        TimeEmbedding = timeEmbedding0.clone()
        if edge_attr is not None:
            edge_attr = (F.relu(self.edge_scale(edge_attr.unsqueeze(-1)))).mean(dim=-1)

        for jj in range(self.nlayers):
            if self.sharedWeights:
                layerIndex = 0
            else:
                layerIndex = jj
            Tstate = F.dropout(Tstate, p=self.dropout, training=self.training)

            Tstate = F.relu(
                self.rescalesBNs[layerIndex](self.rescales[layerIndex](torch.cat([TimeEmbedding, Tstate], dim=-1))))
            Thist = F.relu(
                self.HistBNs[layerIndex](self.HistEmbeds[layerIndex](torch.cat([T0Hist, Thist, Tstate], dim=-1))))

            if self.useMHA:
                mha_out, mha_out_weights = self.mha(acts, acts,
                                                    acts)
                C = torch.log(mha_out_weights * F.relu(self.mha_factor) + 0.0001)
                C = C.mean(dim=0)
                C = 0.5 * (C + C.t())
                C = C[-1, :]
                C = C / C.sum()
                C = C.unsqueeze(-1).unsqueeze(-1)
            else:
                C = self.C[self.order-1]
                C = C / C.sum()
                C = C.unsqueeze(-1).unsqueeze(-1)
            if self.baseline:
                Tstate1 = Tstate
            else:
                Tstate1 = (C * acts).sum(dim=0)
                Tstate1 = Tstate1.squeeze()

            # Spatial Part:
            dState = self.reaction(Tstate1, Thist, T0Hist, layerIndex) - Tstate1
            Tstate = Tstate1 + dState
            Tstate = self.diffusion(Tstate, Tstate, edge_index, layerIndex, explicit=self.explicit, edge_attr=edge_attr)
            acts = torch.cat([acts[1:, :, :], Tstate.unsqueeze(0)], dim=0)

        Tstate = F.dropout(Tstate, p=self.dropoutOC, training=self.training)
        Tstate = self.Kclose(Tstate)

        if regression:
            return Tstate
        return F.log_softmax(Tstate, dim=-1)
