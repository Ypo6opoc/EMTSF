import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.operations_new import V_Package, OPS, G_Package


class Cell_train(nn.Module):

    def __init__(self, args, genotype):

        super().__init__()
        self.args = args
        self.nb_nodes = args.nb_nodes
        self.genotype = genotype
        # self.trans_concat_V = nn.Linear(args.nb_nodes * args.node_dim, args.node_dim, bias = True)
        # self.trans_concat_V = nn.Linear(int(args.nb_nodes/3) * args.node_dim, args.node_dim, bias=True)
        # self.trans_concat_V = nn.Conv2d(in_channels=self.nb_nodes * args.conv_channels, out_channels=args.conv_channels,
        #                                 kernel_size=(1, 1), bias=True)
        self.trans_concat_V = nn.Conv2d(in_channels=int(args.nb_nodes / 3) * args.conv_channels,
                                        out_channels=args.conv_channels,
                                        kernel_size=(1, 1), bias=True)

        # self.batchnorm_V    = nn.BatchNorm1d(args.node_dim)
        self.batchnorm_V = nn.BatchNorm2d(args.conv_channels)
        self.activate = nn.LeakyReLU(args.leaky_slope)
        self.load_genotype()

    def load_genotype(self):
        geno = self.genotype
        link_dict = {}
        module_dict = {}
        for edge in geno['topology']:
            src, dst, ops = edge['src'], edge['dst'], edge['ops']
            dst = f'{dst}'

            if dst not in link_dict:
                link_dict[dst] = []
            link_dict[dst].append(src)

            if dst not in module_dict:
                module_dict[dst] = nn.ModuleList([])
            module_dict[dst].append(V_Package(self.args, OPS[ops](self.args)))

        self.link_dict = link_dict
        self.module_dict = nn.ModuleDict(module_dict)

    def forward(self, input):

        G, V_in = input['G'], input['V']
        states = [V_in]
        for dst in range(1, self.nb_nodes + 1):
            dst = f'{dst}'
            agg = []
            for i, src in enumerate(self.link_dict[dst]):
                sub_input = {'G': G, 'V': states[src], 'V_in': V_in}
                agg.append(self.module_dict[dst][i](sub_input))
            states.append(sum(agg))

        # V = self.trans_concat_V(torch.cat(states[1:], dim = 1))    #把所有节点的输出全算进去了？ 有问题吧
        V = self.trans_concat_V(torch.cat(states[int(-self.nb_nodes / 3):], dim=1))

        if self.batchnorm_V:
            V = self.batchnorm_V(V)

        V = self.activate(V)
        V = F.dropout(V, self.args.dropout, training=self.training)
        V = V + V_in
        return {'G': G, 'V': V}



class ENAS_Cell(nn.Module):

    def __init__(self, args, genotype, c_in,c_out,gdep,dropout,alpha):

        super().__init__()
        self.args = args
        self.nb_nodes = args.nb_nodes
        self.genotype = genotype

        self.mlp = torch.nn.Conv2d((gdep + 1) * c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

        # self.batchnorm_V    = nn.BatchNorm1d(args.node_dim)
        # self.batchnorm_V = nn.BatchNorm2d(args.conv_channels)
        # self.activate = nn.LeakyReLU(args.leaky_slope)
        self.load_genotype()

    def load_genotype(self):
        geno = self.genotype
        link_dict = {}
        module_dict = {}
        for edge in geno['topology']:
            src, dst, ops = edge['src'], edge['dst'], edge['ops']
            dst = f'{dst}'

            if dst not in link_dict:
                link_dict[dst] = []
            link_dict[dst].append(src)

            if dst not in module_dict:
                module_dict[dst] = nn.ModuleList([])
            module_dict[dst].append(G_Package(self.args, OPS[ops](self.args)))

        self.link_dict = link_dict
        self.module_dict = nn.ModuleDict(module_dict)


    def forward(self, V_in, G, adp):
        states = [V_in]
        for dst in range(1, self.nb_nodes + 1):
            dst = f'{dst}'
            agg = []
            for i, src in enumerate(self.link_dict[dst]):
                # sub_input = {'G': G, 'V': states[src], 'V_in': V_in}
                # agg.append(self.module_dict[dst][i](sub_input))
                agg.append(self.module_dict[dst][i](states[src], G, V_in, adp))
            # states.append(sum(agg))
            states.append(self.alpha * V_in + (1 - self.alpha) * sum(agg))

        # V = self.trans_concat_V(torch.cat(states[1:], dim = 1))    #把所有节点的输出全算进去了？ 有问题吧
        # V = self.trans_concat_V(torch.cat(states[int(-self.nb_nodes / 3):], dim=1))
        # for dst in range(1, self.nb_nodes + 1):
        #     states[dst] = self.alpha * V_in + (1 - self.alpha) * states[dst]
        V = torch.cat(states[-int(self.nb_nodes / 3):], dim=1)
        V = torch.cat([V_in, V], dim=1)
        V = self.mlp(V)
        return V


class TC_Cell_train(nn.Module):

    def __init__(self, args, genotype):
        super().__init__()
        self.args = args
        self.genotype = genotype
        self.nb_nodes = args.tc_nodes  # ! warning
        self.cell_arch = genotype
        self.load_genotype()
        # self.trans_concat_TC = nn.Conv2d(in_channels=int(args.tc_nodes) * args.conv_channels,
        #                                 out_channels=args.conv_channels,
        #                                 kernel_size=(1, 1), bias=True)


    def load_genotype(self):
        geno = self.genotype
        link_dict = {}
        module_dict = {}
        self.args.TC_conv_channels = int(self.args.conv_channels / self.args.tc_nodes)
        for edge in geno['topology']:
            src, dst, ops = edge['src'], edge['dst'], edge['ops']
            dst = f'{dst}'

            if dst not in link_dict:
                link_dict[dst] = []
            link_dict[dst].append(src)

            if dst not in module_dict:
                module_dict[dst] = nn.ModuleList([])
            if src==0:
                self.args.TC_residual_channels = self.args.residual_channels
            else:
                self.args.TC_residual_channels = self.args.TC_conv_channels
            module_dict[dst].append(V_Package(self.args, OPS[ops](self.args)))

        self.link_dict = link_dict
        self.module_dict = nn.ModuleDict(module_dict)

    def forward(self, input):
        V_in = input
        states = [V_in]
        for dst in range(1, self.nb_nodes + 1):
            dst = f'{dst}'
            agg = []
            for i, src in enumerate(self.link_dict[dst]):
                sub_input = states[src]
                agg.append(self.module_dict[dst][i](sub_input))
            states.append(sum(agg))
        x = torch.cat(states[int(-self.args.tc_nodes):], dim=1)
        # x = self.trans_concat_TC(x)
        return x


if __name__ == '__main__':
    import yaml 
    from easydict import EasyDict as edict
    geno = yaml.load(open('example_geno.yaml', 'r'))
    geno = geno['Genotype'][0]
    args = edict({
        'nb_nodes': 4,
        'node_dim': 50,
        'leaky_slope': 0.2, 
        'batchnorm_op': True, 
    })
    cell = Cell(args, geno)
    