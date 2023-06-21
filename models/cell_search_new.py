import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from models.operations import OPS
from models.mixed_new import Mixed

'''
cell_arch : 
    topology: list
        (src, dst, weights, ops)
'''


class Cell(nn.Module):

    def __init__(self, args, cell_arch):
        super().__init__()
        self.args = args
        self.nb_nodes = args.nb_nodes * 3  # ! warning
        self.cell_arch = cell_arch
        # self.trans_concat_V = nn.Linear(self.nb_nodes*args.node_dim_gnn, args.node_dim_gnn, bias = True) #有问题
        # self.trans_concat_V = nn.Linear(args.nb_nodes * args.node_dim_gnn, args.node_dim_gnn, bias=True) #我修改的地方
        # self.trans_concat_V = nn.Conv2d(in_channels=self.nb_nodes * args.conv_channels, out_channels=args.conv_channels,
        #                        kernel_size=(1, 1), bias=True)
        self.trans_concat_V = nn.Conv2d(in_channels=args.nb_nodes * args.conv_channels, out_channels=args.conv_channels,
                                        kernel_size=(1, 1), bias=True)

        # self.trans_concat_V = nn.Linear(args.nb_nodes * args.receptive_field, args.receptive_field,
        #                                 bias=True)
        # self.batchnorm_V = nn.BatchNorm1d(args.receptive_field)

        # self.batchnorm_V    = nn.BatchNorm1d(args.node_dim_gnn)
        self.batchnorm_V = nn.BatchNorm2d(args.conv_channels)
        self.activate = nn.LeakyReLU(args.leaky_slope)
        self.load_arch()

    def load_arch(self):
        link_para = {}
        link_dict = {}
        for src, dst, w, ops in self.cell_arch:
            if dst not in link_dict:
                link_dict[dst] = []
            link_dict[dst].append((src, w))
            link_para[str((src, dst))] = Mixed(self.args, ops)

        self.link_dict = link_dict
        self.link_para = nn.ModuleDict(link_para)

    def forward(self, input, weight):
        G, V_in = input['G'], input['V']
        link_para = self.link_para
        link_dict = self.link_dict
        states = [V_in]
        # V_in = V_in.view(-1, self.args.receptive_field)  ###改了
        for dst in range(1, self.nb_nodes + 1):
            tmp_states = []
            for src, w in link_dict[dst]:
                sub_input = {'G': G, 'V': states[src], 'V_in': V_in}
                tmp_states.append(link_para[str((src, dst))](sub_input, weight[w]))
            states.append(sum(tmp_states))

        ## V = self.trans_concat_V(torch.cat(states[1:], dim = 1))   #这个地方感觉错了
        V = self.trans_concat_V(torch.cat(states[int(-self.nb_nodes / 3):], dim=1))
        if self.batchnorm_V:
            V = self.batchnorm_V(V)
        ###新改的
        # V = torch.cat(states[int(-self.nb_nodes/3):], dim=-1)
        # N, C, W, L = states[-1].shape
        # V = V.view(-1, L*int(self.nb_nodes/3))
        # V = self.trans_concat_V(V)

        # if self.batchnorm_V:
        #     V = self.batchnorm_V(V)
        # V = V.view(N, C, W, L)
        ###

        V = self.activate(V)
        # V = F.dropout(V, self.args.dropout, training=self.training)
        # V = V + V_in
        return {'G': G, 'V': V}


class TC_Cell(nn.Module):

    def __init__(self, args, cell_arch):
        super().__init__()
        self.args = args
        self.nb_nodes = args.tc_nodes  # ! warning
        self.cell_arch = cell_arch
        self.load_arch()
        # self.trans_concat_TC = nn.Conv2d(in_channels=int(args.tc_nodes) * args.residual_channels,
        #                                 out_channels=args.conv_channels,
        #                                 kernel_size=(1, 1), bias=True)

    def load_arch(self):
        link_para = {}
        link_dict = {}
        for src, dst, w, ops in self.cell_arch:
            if dst not in link_dict:
                link_dict[dst] = []
            link_dict[dst].append((src, w))
            if src == 0:
                link_para[str((src, dst))] = Mixed(self.args, ops, True)
            else:
                link_para[str((src, dst))] = Mixed(self.args, ops)


        self.link_dict = link_dict
        self.link_para = nn.ModuleDict(link_para)

    def forward(self, input, weight):
        V_in = input
        link_para = self.link_para
        link_dict = self.link_dict
        states = [V_in]
        for dst in range(1, self.args.tc_nodes + 1):
            tmp_states = []
            for src, w in link_dict[dst]:
                sub_input = states[src]
                tmp_states.append(link_para[str((src, dst))](sub_input, weight[w]))
            states.append(sum(tmp_states))
        x = torch.cat(states[int(-self.args.tc_nodes):], dim=1)
        # x = self.trans_concat_TC(x)
        return x

