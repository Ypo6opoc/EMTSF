import torch
import dgl.function as fn
import torch.nn as nn
import numpy as np
# from models.networks import *
from torch.nn.utils import weight_norm

OPS = {
    'V_None': lambda args: V_None(args),
    'V_I': lambda args: V_I(args),
    'V_Max': lambda args: V_Max(args),
    'V_Mean': lambda args: V_Mean(args),
    'V_Min': lambda args: V_Min(args),
    'V_Sum': lambda args: V_Sum(args),
    'V_Sparse': lambda args: V_Sparse(args),
    'V_Dense': lambda args: V_Dense(args),
    'Conv 1x2': lambda args: TimeConv(args, 2),
    'Conv 1x3': lambda args: TimeConv(args, 3),
    'Conv 1x6': lambda args: TimeConv(args, 6),
    'Conv 1x7': lambda args: TimeConv(args, 7),
    'TC_None': lambda args: TC_None(args),
    'TC_I': lambda args: TC_I(args),
}

# First_Stage = ['V_None', 'V_I', 'V_Sparse', 'V_Dense']
# Second_Stage = ['V_I', 'V_Mean', 'V_Sum', 'V_Max']
# # Second_Stage = ['V_Min', 'V_Mean', 'V_Sum', 'V_Max']
# Third_Stage = ['V_None', 'V_I', 'V_Sparse', 'V_Dense']
TC_Stage = ['TC_None', 'Conv 1x2', 'Conv 1x3', 'Conv 1x6', 'Conv 1x7']

First_Stage = ['V_I']
Second_Stage = ['V_Mean']
# Second_Stage = ['V_Min', 'V_Mean', 'V_Sum', 'V_Max']
Third_Stage = ['V_I']


class V_Package(nn.Module):

    def __init__(self, args, operation):

        super().__init__()
        self.args = args
        self.operation = operation
        if type(operation) in [V_None, V_I, TimeConv, TC_I, TC_None]:
            self.seq = None
        else:
            self.seq = nn.Sequential()
            # self.seq.add_module('fc_bn', nn.Linear(args.receptive_field, args.receptive_field, bias=True))
            # self.seq.add_module('fc_bn', nn.Conv2d(in_channels=args.conv_channels, out_channels=args.conv_channels,
            #                    kernel_size=(1, 1), bias=True))
            # self.seq.add_module('fc_bn', nn.Conv2d(in_channels=args.conv_channels, out_channels=args.conv_channels,
            #                                        kernel_size=1, bias=True))

            if args.batchnorm_op:
                # self.seq.add_module('bn', nn.BatchNorm1d(self.args.receptive_field))
                # self.seq.add_module('bn', nn.BatchNorm1d(args.conv_channels))
                self.seq.add_module('bn', nn.BatchNorm2d(args.conv_channels))
            # self.seq.add_module('act', nn.ReLU(inplace=True))

    def forward(self, input):
        V = self.operation(input)
        if self.seq:
            # N, C, W, L = V.shape
            # V = V.view(-1, L)
            V = self.seq(V)
            # V = V.view(N, C, W, L)
        return V


class G_Package(nn.Module):

    def __init__(self, args, operation):
        super().__init__()
        self.args = args
        self.operation = operation

    def forward(self, V, G=None, V_in=None, adp=None):
        if G:
            output = self.operation(V, G, V_in, adp)
        else:
            output = self.operation(V)
        return output


class NodePooling(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.A = nn.Linear(args.receptive_field, args.receptive_field)
        # self.A = nn.Conv2d(in_channels=args.conv_channels, out_channels=args.conv_channels,
        #                        kernel_size=(1, 1), bias=True)
        # self.A = nn.Conv1d(in_channels=args.conv_channels, out_channels=args.conv_channels,
        #                    kernel_size=1, bias=True)
        # self.B        = nn.Linear(args.node_dim_gnn, args.node_dim_gnn)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, V):
        N, C, W, L = V.shape
        V = V.view(-1, L)
        V = self.A(V)
        V = self.activate(V)
        V = V.view(N, C, W, L)

        # V = self.B(V)
        return V


class V_None(nn.Module):

    def __init__(self, args):
        super().__init__()

    def forward(self, V, G, V_in, adp):
        # V = input['V']
        return 0. * V


class V_I(nn.Module):

    def __init__(self, args):
        super().__init__()

    def forward(self, V, G, V_in, adp):
        # V = input['V']
        return V

class TC_None(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cin = args.TC_residual_channels
        self.cout = args.TC_conv_channels
        self.tconv = nn.Conv2d(self.cin, self.cout, (1, 1))
    def forward(self, input):
        B, C, N, L = input.shape
        return torch.zeros((B, self.cout, N, L), requires_grad=False).to(self.args.device)
        # return 0. * self.tconv(input)


class TC_I(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.cin = args.TC_residual_channels
        self.cout = args.TC_conv_channels
        self.tconv = nn.Conv2d(self.cin, self.cout, (1, 1))

    def forward(self, input):
        if self.cout == self.cin:
            return input
        return self.tconv(input)



class V_Max(nn.Module):

    def __init__(self, args):
        super().__init__()
        # self.pooling = NodePooling(args)

    def forward(self, V, G, V_in, adp):
        # G, V = input['G'], input['V']
        # G.ndata['V'] = V
        # G.ndata['V'] = self.pooling(V)
        # G.update_all(fn.copy_u('V', 'M'), fn.max('M', 'V'))
        # return G.ndata['V']
        # V = self.pooling(V)
        G.ndata['V'] = V.permute(2, 0, 1, 3)
        # G.ndata['V'] = V
        # G.update_all(fn.copy_u('V', 'M'), fn.max('M', 'V'))
        G.update_all(fn.u_mul_e('V', 'w', 'M'), fn.max('M', 'V'))
        V = G.ndata['V'].permute(1, 2, 0, 3).contiguous()
        # V = G.ndata['V']
        return V

class V_Mean(nn.Module):

    def __init__(self, args):
        super().__init__()

    def forward(self, V, G, V_in, adp):
        d = adp.sum(1)
        a = adp / d.view(-1, 1)
        x = torch.einsum('ncwl,vw->ncvl',(V,a))
        return x.contiguous()


# class V_Mean(nn.Module):
#
#     def __init__(self, args):
#         super().__init__()
#         # self.pooling = NodePooling(args)
#
#     def forward(self, V, G, V_in, adp):
#         # G, V = input['G'], input['V']
#         # G.ndata['V'] = V
#         # V = self.pooling(V)
#         G.ndata['V'] = V.permute(2, 0, 1, 3)
#         # G.ndata['V'] = V
#         # G.update_all(fn.copy_u('V', 'M'), fn.mean('M', 'V'))
#         G.update_all(fn.u_mul_e('V', 'w', 'M'), fn.mean('M', 'V'))
#         V = G.ndata['V'].permute(1, 2, 0, 3).contiguous()
#         # V = G.ndata['V']
#         return V

class V_Sum(nn.Module):

    def __init__(self, args):
        super().__init__()

    def forward(self, V, G, V_in, adp):
        x = torch.einsum('ncwl,vw->ncvl',(V,adp))
        return x.contiguous()

# class V_Sum(nn.Module):
#
#     def __init__(self, args):
#         super().__init__()
#         # self.pooling = NodePooling(args)
#
#     def forward(self, V, G, V_in, adp):
#         # G, V = input['G'], input['V']
#         # G.ndata['V'] = self.pooling(V)
#         # G.update_all(fn.copy_u('V', 'M'), fn.sum('M', 'V'))
#         # return G.ndata['V']
#
#         # V = self.pooling(V)
#         G.ndata['V'] = V.permute(2, 0, 1, 3)
#         # G.ndata['V'] = V
#         # G.update_all(fn.copy_u('V', 'M'), fn.sum('M', 'V'))
#         G.update_all(fn.u_mul_e('V', 'w', 'M'), fn.sum('M', 'V'))
#         V = G.ndata['V'].permute(1, 2, 0, 3).contiguous()
#         # V = G.ndata['V']
#         return V


class V_Min(nn.Module):

    def __init__(self, args):
        super().__init__()
        # self.pooling = NodePooling(args)

    def forward(self, V, G, V_in, adp):
        # G, V = input['G'], input['V']
        # G.ndata['V'] = self.pooling(V)
        # G.update_all(fn.copy_u('V', 'M'), fn.min('M', 'V'))
        # return G.ndata['V']

        # V = self.pooling(V)
        G.ndata['V'] = V.permute(2, 0, 1, 3)
        # G.ndata['V'] = V
        # G.update_all(fn.copy_u('V', 'M'), fn.min('M', 'V'))
        G.update_all(fn.u_mul_e('V', 'w', 'M'), fn.min('M', 'V'))
        V = G.ndata['V'].permute(1, 2, 0, 3).contiguous()
        # V = G.ndata['V']
        return V


class V_Dense(nn.Module):

    def __init__(self, args):
        super().__init__()
        # self.W = nn.Linear(args.receptive_field * 2, args.receptive_field, bias=True)
        self.W = nn.Conv2d(in_channels=args.conv_channels * 2, out_channels=args.conv_channels,
                           kernel_size=(1, 1), bias=True)
        # self.W = nn.Conv1d(in_channels=args.conv_channels * 2, out_channels=args.conv_channels, kernel_size=1,
        #                    bias=True)

    def forward(self, V, G, V_in, adp):
        # V, V_in = input['V'], input['V_in']
        # N, C, W, L = V.shape
        # V = V.view(-1, L)
        # V_in = V_in.view(-1, L)
        gates = torch.cat([V, V_in], dim=1)
        gates = self.W(gates)
        # return (torch.sigmoid(gates) * V).view(N, C, W, L)
        return torch.sigmoid(gates) * V


class V_Sparse(nn.Module):

    def __init__(self, args):
        super().__init__()
        # self.W = nn.Linear(args.receptive_field * 2, args.receptive_field, bias=True)
        # self.a = nn.Linear(args.receptive_field, 1, bias=False)
        self.activate = nn.ReLU(inplace=True)
        self.W = nn.Conv2d(in_channels=args.conv_channels*2, out_channels=args.conv_channels, kernel_size=(1, 1), bias=True)
        self.a = nn.Conv2d(in_channels=args.conv_channels, out_channels=1, kernel_size=(1, 1), bias=False)

        # self.W = nn.Conv1d(in_channels=args.conv_channels*2, out_channels=args.conv_channels, kernel_size=1, bias=True)
        # self.a = nn.Conv1d(in_channels=args.conv_channels, out_channels=1, kernel_size=args.receptive_field, bias=False)

    def forward(self, V, G, V_in, adp):
        # V, V_in = input['V'], input['V_in']
        # N, C, W, L = V.shape
        # V = V.view(-1, L)
        # V_in = V_in.view(-1, L)
        gates = torch.cat([V, V_in], dim=1)
        # gates = self.W(gates)
        gates = self.activate(self.W(gates))
        gates = self.a(gates)
        # return (torch.sigmoid(gates) * V).view(N, C, W, L)
        return torch.sigmoid(gates) * V


class TimeConv(nn.Module):
    def __init__(self, args, kernal_size, dilation_factor=None):
        super().__init__()
        self.kernal_size = kernal_size
        self.cin = args.TC_residual_channels
        self.cout = args.TC_conv_channels
        if dilation_factor:
            self.dilation_factor = dilation_factor
        else:
            self.dilation_factor = args.curent_dilation_factor
        self.tconv = weight_norm(nn.Conv2d(self.cin, self.cout, (1, self.kernal_size), dilation=(1, self.dilation_factor)))

    def forward(self, input):
        input = nn.functional.pad(input, ((self.kernal_size - 1) * self.dilation_factor, 0, 0, 0))
        x = self.tconv(input)
        return x


if __name__ == '__main__':
    print("test")
