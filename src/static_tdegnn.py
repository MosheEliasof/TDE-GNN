import torch
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, remove_self_loops
import torch.nn as nn
from torch_geometric.utils import get_laplacian



class static_tdegnn(nn.Module):
    def __init__(self, nlayers, nhid, nin, nout, dropout=0.5, h=0.1, sharedWeights=False, multiplicativeReaction=True,
                 addU0=True, dropoutOC=0.5, explicit=False, useReaction=True, useDiffusion=True,
                 useBN=True, lpDiffusion=False, mha_dropout=0.0, useMHA=True, order=None, baseline=False):
        super(static_tdegnn, self).__init__()
        self.order = order if order is not None else nlayers
        self.baseline = baseline
        self.useMHA = useMHA
        self.mha_dropout = mha_dropout
        self.useBN = useBN
        self.useReaction = useReaction
        self.useDiffusion = useDiffusion
        self.explicit = explicit
        self.dropoutOC = dropoutOC
        self.dropout = dropout
        self.multiplicativeReaction = multiplicativeReaction
        self.addU0 = addU0
        self.h = h
        self.lpDiffusion = lpDiffusion
        self.nin = nin
        self.nhid = nhid
        self.nout = nout
        self.nlayers = nlayers
        self.sharedWeights = sharedWeights
        # Open parameters:
        self.Kopen = torch.nn.Linear(nin, nhid)
        self.HistEmbed_conv1d = torch.nn.Conv1d(in_channels=nhid, out_channels=nhid, kernel_size=1)


        # Reaction parameters
        self.KR1 = torch.nn.ModuleList()
        self.KR2 = torch.nn.ModuleList()

        self.bns = torch.nn.ModuleList()
        self.bns2 = torch.nn.ModuleList()
        self.bnsOpen = torch.nn.BatchNorm1d(nhid)
        self.bnsOpen_acts = torch.nn.BatchNorm1d(nhid)

        if self.addU0:
            self.KRU0_1 = torch.nn.ModuleList()

        for i in range(self.nlayers):
            self.bns.append(torch.nn.BatchNorm1d(nhid))
            self.bns2.append(torch.nn.BatchNorm1d(nhid))

            self.KR1.append(torch.nn.Linear(nhid, 1 * nhid))
            self.KR2.append(torch.nn.Linear(1 * nhid, nhid))
            if self.addU0:
                self.KRU0_1.append(torch.nn.Linear(nhid, nhid))

        # Diffusion coefficients:
        self.Kappa = torch.nn.ParameterList()
        for i in range(self.nlayers):
            self.Kappa.append(torch.nn.Parameter(torch.ones(nhid)))
        # Closing parameters:
        self.Kclose = torch.nn.Linear(nhid, nout)

        # Create optimization groups:
        # Create reaction groups:
        self.reactionParams = torch.nn.ModuleList()
        self.reactionParams.append(self.bns)
        self.reactionParams.append(self.bns2)
        self.reactionParams.append(self.KR1)
        self.reactionParams.append(self.KR2)
        if self.addU0:
            self.reactionParams.append(self.KRU0_1)

        # Create diffusion groups:
        self.diffusionParams = torch.nn.ModuleList()
        self.diffusionParams.append(self.Kappa)

        # Create open close group:
        self.projParams = torch.nn.ModuleList()
        self.projParams.append(self.bnsOpen)
        self.projParams.append(self.Kopen)
        self.projParams.append(self.Kclose)
        self.projParams.append(self.HistEmbed_conv1d)
        self.projParams.append(self.bnsOpen_acts)
        self.mha = torch.nn.MultiheadAttention(embed_dim=nhid, num_heads=1, dropout=self.mha_dropout)
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
                1e-2 * torch.randn(
                    self.KR2[i].weight.shape))
            self.KR1[i].bias = torch.nn.Parameter(torch.zeros_like(self.KR1[i].bias))
            self.KR2[i].bias = torch.nn.Parameter(torch.zeros_like(self.KR2[i].bias))

            self.mha_factor = torch.nn.Parameter(torch.Tensor([10]))

            if self.addU0:
                self.KRU0_1[i].weight = torch.nn.Parameter(
                    torch.eye(self.KRU0_1[i].weight.shape[0], self.KRU0_1[i].weight.shape[1]) + 1e-2 * torch.randn(
                        self.KRU0_1[i].weight.shape))
                self.KRU0_1[i].bias = torch.nn.Parameter(torch.zeros_like(self.KRU0_1[i].bias))

        glorot(self.Kopen.weight)
        glorot(self.Kclose.weight)
        zeros(self.Kopen.bias)
        zeros(self.Kclose.bias)



    def reaction(self, T, T0, layer, residual=False):
        dT = self.KR1[layer](T)
        if self.addU0:
            dT += self.KRU0_1[layer](T0)
        if self.multiplicativeReaction:
            dT2 = self.KR2[layer](T)
            dT = dT + (T * F.hardtanh(dT2))
        T = (T + self.h * dT)
        if self.useBN:
            T = self.bns[layer](T)
        T = F.elu(T)
        return T

    def diffusion(self, T, edge_index, layer, explicit, edge_attr=None):
        cgiter = 10
        Kd = self.Kappa[layer]
        Kd = F.hardtanh(Kd, min_val=0, max_val=1)
        if explicit:
            from torch_geometric.nn.conv.gcn_conv import gcn_norm
            lap_edge_index, lap_edge_weight = get_laplacian(edge_index, edge_weight=edge_attr, normalization='sym',
                                                            num_nodes=T.shape[0])
            prop_edge_index, prop_edge_weight = gcn_norm(edge_index, edge_weight=edge_attr, num_nodes=T.shape[0],
                                                         improved=False)

            lap_op = torch.sparse_coo_tensor(indices=lap_edge_index, values=lap_edge_weight,
                                             size=(T.shape[0], T.shape[0]))
            prop_op = torch.sparse_coo_tensor(indices=prop_edge_index, values=prop_edge_weight,
                                              size=(T.shape[0], T.shape[0]))

            if self.lpDiffusion:
                residual_diffuse = (T - torch.sparse.mm(prop_op, T)) * Kd
                tmp = (T - self.h * residual_diffuse)
                return tmp
            else:
                LapT = torch.sparse.mm(lap_op, T)
                LapT = LapT * Kd
                out = T - self.h * (LapT)
            return out

        def matvec(Y, edge_index, h=self.h):
            lap_edge_index, lap_edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=Y.shape[0])
            lap_op = torch.sparse_coo_tensor(indices=lap_edge_index, values=lap_edge_weight,
                                             size=(Y.shape[0], Y.shape[0]))

            LapY = torch.sparse.mm(lap_op, Y)
            LapY = LapY * Kd
            out = Y + h * (LapY)
            return out

        X = T.clone()
        R = T.clone()
        P = R.clone()
        normRold = R.norm() ** 2
        for i in range(cgiter):
            LP = matvec(P, edge_index)
            alpha = R.norm() ** 2 / (torch.sum(P * LP) + 1e-6)
            X = X + alpha * P
            R = R - alpha * LP
            normR = R.norm() ** 2
            if normR < 1e-5:
                break
            beta = normR / (normRold + 1e-6)
            normRold = normR.clone()
            P = R + beta * P
        return X

    def forward(self, T, edge_index, regression=False):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=T.shape[0])
        #T = F.dropout(T, p=self.dropoutOC, training=self.training)
        T = (self.Kopen(T))
        if self.useBN:
            T = self.bnsOpen(T)
        T = F.elu(T)
        T0orig = T.clone()
        acts = T0orig.clone().unsqueeze(0)  # 1 x N X C
        acts = torch.repeat_interleave(acts, repeats=self.order, dim=0).permute(1, 2, 0)
        acts = F.relu(self.bnsOpen_acts(self.HistEmbed_conv1d(acts)))
        acts = acts.permute(2, 0, 1)

        for jj in range(self.nlayers):
            if self.sharedWeights:
                layerIndex = 0
            else:
                layerIndex = jj
            T0 = T0orig
            if self.addU0:
                T0 = F.dropout(T0orig.clone(), p=self.dropout, training=self.training)
            if self.addU0:
                T0 = F.dropout(T0orig.clone(), p=self.dropout, training=self.training)
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
                C = self.C[self.order - 1]
                C = C / C.sum()
                C = C.unsqueeze(-1).unsqueeze(-1)

            if self.baseline:
                T1 = T
            else:
                T1 = (C * acts).sum(dim=0)
                T1 = T1.squeeze()
            T = self.reaction(T1, T0, layerIndex)
            T = self.diffusion(T, edge_index, layerIndex, explicit=self.explicit) if self.useDiffusion else T
            acts = torch.cat([acts[1:, :, :], T.unsqueeze(0)], dim=0)

        T = F.dropout(T, p=self.dropoutOC, training=self.training)
        T = self.Kclose(T)
        return T
        if regression:
            return T
        return F.log_softmax(T, dim=-1)
