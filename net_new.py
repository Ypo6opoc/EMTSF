from layer import *

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
new = True
if new:
    from models.cell_search_new import Cell, TC_Cell
    from models.operations_new import OPS, First_Stage, Second_Stage, Third_Stage, TC_Stage
    from models.cell_train_new import Cell_train, TC_Cell_train, ENAS_Cell
else:
    from models.cell_search import Cell
    from models.operations import OPS, First_Stage, Second_Stage, Third_Stage
    from models.cell_train import Cell_train
from models.networks import MLP
# from data import TransInput, TransOutput, get_trans_input

# from models.graph import NodeFeaExtractor, Dynamic_NodeFeaExtractor

import networkx as nx
import scipy.sparse as sp


import dgl.function as fn
class My_layer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, V, G):
        G.ndata['V'] = V.permute(2, 0, 1, 3)
        G.update_all(fn.u_mul_e('V', 'w', 'M'), fn.sum('M', 'V'))
        V = G.ndata['V'].permute(1, 2, 0, 3).contiguous()
        return V

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, adj):
        # d = adj.sum(1)
        # a = adj / d.view(-1, 1)
        a = adj
        x = torch.einsum('ncwl,vw->ncvl',(x,a))
        return x.contiguous()

class My_GNN(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(My_GNN, self).__init__()
        # self.nconv = nconv()
        self.nconv = My_layer()
        self.nconv_old = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, V, G, adj):
        h = V
        out = [h]
        for i in range(self.gdep):
            # c1 = self.nconv(h, G)
            # c2 = self.nconv_old(h, adj)
            # htest = self.alpha * V + (1 - self.alpha) * self.nconv_old(h, adj)

            #
            # adj_matrix = G.adjacency_matrix(transpose=False)  # 转换为稀疏矩阵格式
            # adj_matrix = adj_matrix.to_dense().to('cuda:0')  # 转换为稠密矩阵格式
            # adj_matrix[G.edges()[0].long(), G.edges()[1].long()] = G.edata['w']
            # hamud = (adj == adj_matrix).all()
            h = self.alpha * V + (1 - self.alpha) * self.nconv_old(h, adj)
            # h = self.alpha * V.double() + (1 - self.alpha) * self.nconv(h.double(), G)
            # h = self.alpha * V + (1 - self.alpha) * self.nconv(h, G)
            # G.edata['w'] = G.edata['w'].double()
            # h64 = self.alpha * V.double() + (1 - self.alpha) * self.nconv(V.double(), G)
            # hamud = (h == htest).all()
            #
            # hamud1 = (c1 == c2)
            # h = self.alpha * V + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho

class train_net(nn.Module):
    def __init__(self, args, genotypes, TC_genotypes, gcn_true, buildA_true, gcn_depth, num_nodes, device, loss_fn, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(train_net, self).__init__()
        #GNAS-MP模块
        self.args = args
        self.nb_layers = layers*args.GNN_depth_per_block
        self.genotypes = genotypes
        self.TC_genotypes = TC_genotypes
        # self.cell_arch_topo = self.load_cell_arch()  # obtain architecture topology
        # self.cell_arch_para = self.init_cell_arch_para()  # register architecture topology parameters
        # self.cells = nn.ModuleList([Cell_train(args, genotypes[i]) for i in range(self.nb_layers)])
        # self.cells = nn.ModuleList([Cell(args, self.cell_arch_topo[i]) for i in range(self.nb_layers)])
        depth = 0
        self.cells = nn.ModuleList()
        self.cells_reverse = nn.ModuleList()
        self.TC_cells_filter = nn.ModuleList()
        self.TC_cells_gate = nn.ModuleList()
        self.gnn_mlp = nn.ModuleList()
        self.propalpha =propalpha
        self.loss_fn = loss_fn
        # self.trans_input_fn = trans_input_fn
        # self.trans_input = TransInput(trans_input_fn)
        # self.trans_output = TransOutput(args)

        # self.stfea_encode = NodeFeaExtractor(self.args.st_embedding_dim, self.args.fc_dim)
        # self.dyfea_encode = Dynamic_NodeFeaExtractor(int(self.args.st_embedding_dim / 2), int(max(self.seq_length,self.receptive_field)*self.args.num_nodes))  ###
        self.static_feat = static_feat

        self.activate=nn.ReLU(True)
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        # self.filter_convs = nn.ModuleList()
        # self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        # self.gconv1 = nn.ModuleList()
        # self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        # self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                # self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                # self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))

                ###TC_test_2022
                rf_size_j = 1
                args.curent_dilation_factor = new_dilation
                self.TC_cells_filter.append(TC_Cell_train(args, self.TC_genotypes[j-1]))
                self.TC_cells_gate.append(TC_Cell_train(args, self.TC_genotypes[j-1]))
                ###

                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    ##  MY_GNN_NAS
                    if self.seq_length > self.receptive_field:
                        args.receptive_field = self.seq_length-rf_size_j+1
                    else:
                        args.receptive_field = self.receptive_field-rf_size_j+1
                    args.conv_channels = conv_channels
                    for _ in range(self.args.GNN_depth_per_block):
                        self.cells.append(ENAS_Cell(args, self.genotypes[depth], conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                        self.cells_reverse.append(ENAS_Cell(args, self.genotypes[depth], conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                        depth += 1
                    # self.gnn_mlp.append(nn.Conv2d(in_channels=conv_channels*(self.args.GNN_depth_per_block+1), out_channels=residual_channels, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True))
                    # self.gnn_mlp_reverse.append(nn.Conv2d(in_channels=conv_channels*(self.args.GNN_depth_per_block+1), out_channels=residual_channels, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True))
                    ##


                    # self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))   #添加GNN
                    # self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        # self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
        #                                      out_channels=end_channels,
        #                                      kernel_size=(1,1),
        #                                      bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)


        self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels,
                               kernel_size=(1, max(self.seq_length, self.receptive_field)), bias=True)

        self.idx = torch.arange(self.num_nodes).to(device)
        # self.arch_para_dict = self.group_arch_parameters()
        ########################################################
        self.input_dim = 1
        self.node_emb_dim, self.temp_dim_tid, self.temp_dim_diw = 32, 32, 32
        # spatial embeddings
        # self.node_emb = nn.Embedding(self.num_nodes, node_dim)
        self.node_emb1 = nn.Parameter(
            torch.empty(self.num_nodes, self.node_emb_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.node_emb1)
        self.node_emb2 = nn.Parameter(
            torch.empty(self.num_nodes, self.node_emb_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.node_emb2)
        # temporal embeddings
        if args.add_time_in_day:
            # self.time_in_day_emb = nn.Embedding(args.steps_per_day, self.temp_dim_tid)
            self.input_dim += 1
            self.time_in_day_emb = nn.Parameter(
                torch.empty(args.steps_per_day, self.temp_dim_tid), requires_grad=True)
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if args.add_day_in_week:
            # self.day_in_week_emb = nn.Embedding(7, self.temp_dim_diw)
            self.input_dim += 1
            self.day_in_week_emb = nn.Parameter(
                torch.empty(7, self.temp_dim_diw), requires_grad=True)
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        # self.time_series_emb_layer = nn.Conv2d(
        #     in_channels=self.input_dim * self.seq_length, out_channels=self.node_emb_dim, kernel_size=(1, 1), bias=True)
        # final_channels = skip_channels * 1 + self.node_emb_dim + args.add_day_in_week * self.temp_dim_diw + args.add_time_in_day * self.temp_dim_tid
        # self.end_conv_0 = nn.Conv2d(in_channels=final_channels,
        #                             out_channels=out_dim,
        #                             kernel_size=(1, 1),
        #                             bias=True)
        final_channels = args.add_day_in_week * self.temp_dim_diw + args.add_time_in_day * self.temp_dim_tid
        self.end_conv_0 = nn.Conv2d(in_channels=final_channels,
                                    out_channels=skip_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_01 = nn.Conv2d(in_channels=skip_channels*2,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(final_channels, final_channels) for _ in range(2)])

        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha,
                                    static_feat=static_feat)

    def forward(self, input, idx=None):
        # seq_len = input.size(3)
        # assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'
        #
        # if self.seq_length<self.receptive_field:
        #     input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))
        #

        seq_len = input.size(1)
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'

        # if self.seq_length < self.receptive_field:
        #     input = nn.functional.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))

        ################################################################
        # prepare data
        input_data = input[..., range(self.input_dim)]

        if self.args.add_time_in_day:
            t_i_d_data = input[..., 1]
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * self.args.steps_per_day).type(torch.LongTensor)]
        else:
            time_in_day_emb = None
        if self.args.add_day_in_week:
            d_i_w_data = input[..., -1]
            day_in_week_emb = self.day_in_week_emb[(
                d_i_w_data[:, -1, :]).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        # time_series_emb = self.time_series_emb_layer(input_data)

        node_emb = []

        # expand node embeddings
        node_emb.append(self.node_emb1.unsqueeze(0).expand(
            batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
        node_emb.append(self.node_emb2.unsqueeze(0).expand(
            batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        # concate all embeddings
        # hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)
        ################################################################
        input = torch.unsqueeze(input[..., 0], dim=1)
        input = input.transpose(2, 3)

        # st_node_fea = self.stfea_encode(self.static_feat)
        st_node_fea = self.node_emb1
        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    graph, adp, graph_reverse = self.gc(self.idx, None, None)
                else:
                    graph, adp, graph_reverse = self.gc(idx, None, None)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            # ###
            # filter = self.filter_convs[i](x)
            # filter = torch.tanh(filter)
            # gate = self.gate_convs[i](x)
            # gate = torch.sigmoid(gate)
            # x = filter * gate
            # ##
            ###
            #
            filter = self.TC_forward(x, i, True)
            filter = torch.tanh(filter)
            gate = self.TC_forward(x, i, False)
            gate = torch.sigmoid(gate)
            x = filter * gate


            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:

                # G, V = graph, x
                x = self.cells[i](x, graph, adp.transpose(1, 0)) + self.cells_reverse[i](x, graph_reverse, adp)
                ###

                # x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        #################################
        # hidden = torch.cat([x] + node_emb + tem_emb, dim=1)
        # hidden = self.encoder(hidden)
        # hidden = self.end_conv_0(hidden)
        # return hidden

        hidden = torch.cat(tem_emb, dim=1)
        hidden = self.encoder(hidden)
        hidden = self.end_conv_0(F.dropout(hidden, self.dropout, training=self.training))
        x = torch.cat([x]+[hidden], dim=1)
        x = F.relu(self.end_conv_01(x))
        x = self.end_conv_2(x)
        return x
        #################################
        # x = F.relu(self.end_conv_1(x))
        # x = self.end_conv_2(x)
        # return x


###GNAS-MP
    def GNN_forward(self, input, index):

        # input = self.trans_input(input)
        G, V = input['G'], input['V']
        # if self.args.pos_encode > 0:
        #     V = V + self.position_encoding(G.ndata['pos_enc'].float().cuda())
        output = {'G': G, 'V': V}
        out = [V]
        # for i, cell in enumerate(self.cells):
        #     output = cell(output, arch_para_dict[i])
        for i in range(self.args.GNN_depth_per_block):
            output = self.cells[index*2+i](output)
            output['V'] = self.propalpha * V + (1 - self.propalpha) * output['V']
            out.append(output['V'])

        ho = torch.cat(out, dim=1)
        ho = self.gnn_mlp[index](ho)
        return ho

    def TC_forward(self, input, index, is_filter=True):
        if is_filter:
            output = self.TC_cells_filter[index](input)
        else:
            output = self.TC_cells_gate[index](input)
        return output

    def load_cell_arch(self):
        cell_arch_topo = []
        for _ in range(self.nb_layers):
            arch_topo = self.load_cell_arch_by_layer()
            cell_arch_topo.append(arch_topo)
        return cell_arch_topo

    def load_cell_arch_by_layer(self):
        arch_topo = []
        w = 0
        for dst in range(1, self.args.nb_nodes + 1):
            for src in range(dst):
                arch_topo.append((src, dst, w, First_Stage))
                w += 1
        for dst in range(self.args.nb_nodes + 1, 2 * self.args.nb_nodes + 1):
            src = dst - self.args.nb_nodes
            arch_topo.append((src, dst, w, Second_Stage))
            w += 1
        for dst in range(2 * self.args.nb_nodes + 1, 3 * self.args.nb_nodes + 1):
            for src in range(self.args.nb_nodes + 1, dst):
                arch_topo.append((src, dst, w, Third_Stage))
                w += 1
        return arch_topo

    def init_cell_arch_para(self):
        cell_arch_para = []
        for i_layer in range(self.nb_layers):
            arch_para = self.init_arch_para(self.cell_arch_topo[i_layer])
            cell_arch_para.extend(arch_para)
            self.nb_cell_topo = len(arch_para)
        return cell_arch_para

    def init_arch_para(self, arch_topo):
        arch_para = []
        for src, dst, w, ops in arch_topo:
            arch_para.append(Variable(1e-3 * torch.rand(len(ops)).cuda(), requires_grad=True))
        return arch_para

    def group_arch_parameters(self):
        group = []
        start = 0
        for _ in range(self.nb_layers):
            group.append(self.arch_parameters()[start: start + self.nb_cell_topo])
            start += self.nb_cell_topo
        return group

    def new(self):
        model_new = Model_Search(self.args, get_trans_input(self.args), self.loss_fn).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def load_alpha(self, alphas):
        for x, y in zip(self.arch_parameters(), alphas):
            x.data.copy_(y.data)

    def arch_parameters(self):
        return self.cell_arch_para

    def _loss(self, input, targets, dataset):
        scores = self.forward(input)
        scores = torch.squeeze(scores)
        scale = dataset.scale.expand(scores.size(0), dataset.m)
        return self.loss_fn(scores * scale, targets * scale)


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden


class train_net_multi(nn.Module):
    def __init__(self, args, genotypes, TC_genotypes, gcn_true, buildA_true, gcn_depth, num_nodes, device, loss_fn, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(train_net_multi, self).__init__()
        self.args = args
        self.nb_layers = layers*args.GNN_depth_per_block
        self.genotypes = genotypes
        self.TC_genotypes = TC_genotypes
        # self.cell_arch_topo = self.load_cell_arch()  # obtain architecture topology
        # self.cell_arch_para = self.init_cell_arch_para()  # register architecture topology parameters
        # self.cells = nn.ModuleList([Cell_train(args, genotypes[i]) for i in range(self.nb_layers)])
        # self.cells = nn.ModuleList([Cell(args, self.cell_arch_topo[i]) for i in range(self.nb_layers)])
        depth = 0
        self.cells = nn.ModuleList()
        self.cells_reverse = nn.ModuleList()
        self.TC_cells_filter = nn.ModuleList()
        self.TC_cells_gate = nn.ModuleList()
        self.gnn_mlp = nn.ModuleList()
        self.propalpha =propalpha
        # self.loss_fn = loss_fn
        # self.trans_input_fn = trans_input_fn
        # self.trans_input = TransInput(trans_input_fn)
        # self.trans_output = TransOutput(args)

        # self.stfea_encode = NodeFeaExtractor(self.args.st_embedding_dim, self.args.fc_dim)
        # self.dyfea_encode = Dynamic_NodeFeaExtractor(int(self.args.st_embedding_dim / 2), int(max(self.seq_length,self.receptive_field)*self.args.num_nodes))  ###
        self.static_feat = static_feat

        self.activate=nn.ReLU(True)
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        # self.filter_convs = nn.ModuleList()
        # self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        # self.gconv1 = nn.ModuleList()
        # self.gconv2 = nn.ModuleList()
        # self.mygconv1 = nn.ModuleList()  ###
        # self.mygconv2 = nn.ModuleList()  ###
        self.norm = nn.ModuleList()
        self.in_dim = in_dim
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        # self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                # self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                # self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))


                rf_size_j = 1
                args.curent_dilation_factor = new_dilation
                self.TC_cells_filter.append(TC_Cell_train(args, self.TC_genotypes[j-1]))
                self.TC_cells_gate.append(TC_Cell_train(args, self.TC_genotypes[j-1]))
                ###

                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))

                self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                out_channels=skip_channels,
                                                kernel_size=(1, self.seq_length-rf_size_j+1)))

                if self.gcn_true:

                    if self.seq_length > self.receptive_field:
                        args.receptive_field = self.seq_length-rf_size_j+1
                    else:
                        args.receptive_field = self.receptive_field-rf_size_j+1
                    args.conv_channels = conv_channels
                    for _ in range(self.args.GNN_depth_per_block):
                        self.cells.append(ENAS_Cell(args, self.genotypes[depth], conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                        self.cells_reverse.append(ENAS_Cell(args, self.genotypes[depth], conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                        depth += 1
                    # self.gnn_mlp.append(nn.Conv2d(in_channels=conv_channels*(self.args.GNN_depth_per_block+1), out_channels=residual_channels, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True))
                    # self.gnn_mlp_reverse.append(nn.Conv2d(in_channels=conv_channels*(self.args.GNN_depth_per_block+1), out_channels=residual_channels, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True))
                    ##


                    # self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))   #添加GNN
                    # self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                    # self.mygconv1.append(My_GNN(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    # self.mygconv2.append(My_GNN(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        # self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
        #                                      out_channels=end_channels,
        #                                      kernel_size=(1,1),
        #                                      bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)

        self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
        self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)




        self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels,
                               kernel_size=(1, self.seq_length), bias=True)

        self.idx = torch.arange(self.num_nodes).to(device)
        # self.arch_para_dict = self.group_arch_parameters()
        ########################################################
        self.input_dim = 1
        self.node_emb_dim, self.temp_dim_tid, self.temp_dim_diw = 32, 32, 32
        # spatial embeddings
        # self.node_emb = nn.Embedding(self.num_nodes, node_dim)

        # temporal embeddings
        if args.add_time_in_day:
            # self.time_in_day_emb = nn.Embedding(args.steps_per_day, self.temp_dim_tid)
            self.input_dim += 1
            self.time_in_day_emb = nn.Parameter(
                torch.empty(args.steps_per_day, self.temp_dim_tid), requires_grad=True)
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if args.add_day_in_week:
            # self.day_in_week_emb = nn.Embedding(7, self.temp_dim_diw)
            self.input_dim += 1
            self.day_in_week_emb = nn.Parameter(
                torch.empty(7, self.temp_dim_diw), requires_grad=True)
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        # self.time_series_emb_layer = nn.Conv2d(
        #     in_channels=self.input_dim * self.seq_length, out_channels=self.node_emb_dim, kernel_size=(1, 1), bias=True)
        # final_channels = skip_channels * 1 + self.node_emb_dim + args.add_day_in_week * self.temp_dim_diw + args.add_time_in_day * self.temp_dim_tid
        # self.end_conv_0 = nn.Conv2d(in_channels=final_channels,
        #                             out_channels=out_dim,
        #                             kernel_size=(1, 1),
        #                             bias=True)
        final_channels = args.add_day_in_week * self.temp_dim_diw + args.add_time_in_day * self.temp_dim_tid
        self.end_conv_0 = nn.Conv2d(in_channels=final_channels,
                                    out_channels=skip_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_01 = nn.Conv2d(in_channels=skip_channels*2,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(final_channels, final_channels) for _ in range(2)])

        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha,
                                    static_feat=static_feat)

    def forward(self, input, idx=None):
        # seq_len = input.size(3)
        # assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'
        #
        # if self.seq_length<self.receptive_field:
        #     input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))
        #

        seq_len = input.size(1)
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'

        # if self.seq_length < self.receptive_field:
        #     input = nn.functional.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))

        ################################################################
        # prepare data
        input_data = input[..., range(self.input_dim)]

        if self.args.add_time_in_day:
            t_i_d_data = input[..., 1]
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * self.args.steps_per_day).type(torch.LongTensor)]
        else:
            time_in_day_emb = None
        if self.args.add_day_in_week:
            d_i_w_data = input[..., -1]
            day_in_week_emb = self.day_in_week_emb[(
                d_i_w_data[:, -1, :]).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        # time_series_emb = self.time_series_emb_layer(input_data)

        node_emb = []

        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        # concate all embeddings
        # hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)
        ################################################################
        # input = torch.unsqueeze(input[..., 0:self.in_dim], dim=1)
        # input = input.transpose(2, 3)
        input = input.transpose(1, 3)
        input = input[:, :self.in_dim, :, :]
        # st_node_fea = self.stfea_encode(self.static_feat)

        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    graph, adp, graph_reverse = self.gc(self.idx, None, None)
                else:
                    graph, adp, graph_reverse = self.gc(idx, None, None)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            # ###
            # filter = self.filter_convs[i](x)
            # filter = torch.tanh(filter)
            # gate = self.gate_convs[i](x)
            # gate = torch.sigmoid(gate)
            # x = filter * gate
            ##
            #
            filter = self.TC_forward(x, i, True)
            filter = torch.tanh(filter)
            gate = self.TC_forward(x, i, False)
            gate = torch.sigmoid(gate)
            x = filter * gate


            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:

                ###
                # G, V = graph, x
                x = self.cells[i](x, graph, adp.transpose(1, 0)) + self.cells_reverse[i](x, graph_reverse, adp) ###我的GNN计算
                ###
                # x = self.mygconv1[i](x, graph, adj) + self.mygconv2[i](x, graph, adj.transpose(1,0))
                # x = self.mygconv1[i](x, graph_reverse, adp) + self.mygconv2[i](x, graph, adp.transpose(1, 0))
                # x = self.gconv1[i](x, adp.transpose(1, 0))+self.gconv2[i](x, adp)        ###GNN运算
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        #################################
        # hidden = torch.cat([x] + node_emb + tem_emb, dim=1)
        # hidden = self.encoder(hidden)
        # hidden = self.end_conv_0(hidden)
        # return hidden

        hidden = torch.cat(tem_emb, dim=1)
        hidden = self.encoder(hidden)
        hidden = self.end_conv_0(F.dropout(hidden, self.dropout, training=self.training))
        x = torch.cat([x]+[hidden], dim=1)
        x = F.relu(self.end_conv_01(x))
        x = self.end_conv_2(x)
        return x
        #################################
        #x = F.relu(self.end_conv_1(x))
        #x = self.end_conv_2(x)
        #return x


###GNAS-MP
    def GNN_forward(self, input, index):

        # input = self.trans_input(input)
        G, V = input['G'], input['V']
        # if self.args.pos_encode > 0:
        #     V = V + self.position_encoding(G.ndata['pos_enc'].float().cuda())
        output = {'G': G, 'V': V}
        out = [V]
        # for i, cell in enumerate(self.cells):
        #     output = cell(output, arch_para_dict[i])
        for i in range(self.args.GNN_depth_per_block):
            output = self.cells[index*2+i](output)
            output['V'] = self.propalpha * V + (1 - self.propalpha) * output['V']
            out.append(output['V'])

        ho = torch.cat(out, dim=1)
        ho = self.gnn_mlp[index](ho)
        return ho

    def TC_forward(self, input, index, is_filter=True):
        if is_filter:
            output = self.TC_cells_filter[index](input)
        else:
            output = self.TC_cells_gate[index](input)
        return output

    def load_cell_arch(self):
        cell_arch_topo = []
        for _ in range(self.nb_layers):
            arch_topo = self.load_cell_arch_by_layer()
            cell_arch_topo.append(arch_topo)
        return cell_arch_topo

    def load_cell_arch_by_layer(self):
        arch_topo = []
        w = 0
        for dst in range(1, self.args.nb_nodes + 1):
            for src in range(dst):
                arch_topo.append((src, dst, w, First_Stage))
                w += 1
        for dst in range(self.args.nb_nodes + 1, 2 * self.args.nb_nodes + 1):
            src = dst - self.args.nb_nodes
            arch_topo.append((src, dst, w, Second_Stage))
            w += 1
        for dst in range(2 * self.args.nb_nodes + 1, 3 * self.args.nb_nodes + 1):
            for src in range(self.args.nb_nodes + 1, dst):    ##这里我改了
                arch_topo.append((src, dst, w, Third_Stage))
                w += 1
        return arch_topo

    def init_cell_arch_para(self):
        cell_arch_para = []
        for i_layer in range(self.nb_layers):
            arch_para = self.init_arch_para(self.cell_arch_topo[i_layer])
            cell_arch_para.extend(arch_para)
            self.nb_cell_topo = len(arch_para)
        return cell_arch_para

    def init_arch_para(self, arch_topo):
        arch_para = []
        for src, dst, w, ops in arch_topo:
            arch_para.append(Variable(1e-3 * torch.rand(len(ops)).cuda(), requires_grad=True))
        return arch_para

    def group_arch_parameters(self):
        group = []
        start = 0
        for _ in range(self.nb_layers):
            group.append(self.arch_parameters()[start: start + self.nb_cell_topo])
            start += self.nb_cell_topo
        return group

    # def new(self):
    #     model_new = Model_Search(self.args, get_trans_input(self.args), self.loss_fn).cuda()
    #     for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
    #         x.data.copy_(y.data)
    #     return model_new

    def load_alpha(self, alphas):
        for x, y in zip(self.arch_parameters(), alphas):
            x.data.copy_(y.data)

    def arch_parameters(self):
        return self.cell_arch_para

    def _loss(self, input, targets, dataset):
        scores = self.forward(input)
        scores = torch.squeeze(scores)
        scale = dataset.scale.expand(scores.size(0), dataset.m)
        print("error")
        # return self.loss_fn(scores * scale, targets * scale)
        return None
