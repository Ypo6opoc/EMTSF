"""
import os
import sys
import dgl
import yaml
import time
import torch
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from datetime import datetime
from tqdm import tqdm
# from models.model_train import *
from utils.utils import *
# from tensorboardX import SummaryWriter
# from utils.record_utils import record_run

import torch.nn as nn
# os.environ['CUDA_VISIBLE_DEVICES']='0'
new = True
if new:
    from net_new import train_net
else:
    from net import train_net

# from util import *
from util_new import *

import warnings
warnings.filterwarnings('ignore')

class Trainer(object):

    def __init__(self, args):


        self.args = args
        device = 'cuda:0'
        device = torch.device(args.device)
        cudnn.benchmark = True
        cudnn.enabled = True

        if args.L1Loss:
            self.loss_fn = nn.L1Loss(size_average=False).to(device)
        else:
            self.loss_fn = nn.MSELoss(size_average=False).to(device)
        self.evaluateL2 = nn.MSELoss(size_average=False).to(device)
        self.evaluateL1 = nn.L1Loss(size_average=False).to(device)


        #cuda4:
        if not os.path.isfile(args.load_genotypes):
            raise Exception('Genotype file not found!')
        else:
            with open(args.load_genotypes, "r") as f:
                genotypes      = yaml.load(f, Loader=yaml.FullLoader)
                args.nb_layers = len(genotypes['GC_Genotype'])
                args.nb_nodes  = len({ edge['dst'] for edge in genotypes['GC_Genotype'][0]['topology']})
                args.tc_nodes  = len({edge['dst'] for edge in genotypes['TC_Genotype'][0]['topology']})

        static_feat = None
        self.model = train_net(args, genotypes['GC_Genotype'], genotypes['TC_Genotype'], args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                           device, loss_fn=self.loss_fn, dropout=args.dropout, subgraph_size=args.subgraph_size,
                           node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                           conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                           skip_channels=args.skip_channels, end_channels=args.end_channels,
                           seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                           layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha,
                           layer_norm_affline=False, static_feat=static_feat)
        self.model = self.model.to(device)


        Data = DataLoaderS(args, args.data, 0.6, 0.2, device, args.horizon, args.seq_in_len, args.normalize)
        self.dataset = Data

        self.search_data = self.dataset.train
        self.val_data = self.dataset.valid
        self.test_data = self.dataset.test

        from trainer import Optim
        self.optimizer = Optim(
            self.model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer  = self.optimizer.optimizer,
            T_max      = float(args.epochs),
            eta_min    = args.lr_min
        )


    def run(self):
        
        self.log_record(f'=> Train Genotypes')
        self.lr = self.args.lr
        best_val = 100
        for i_epoch in range(self.args.epochs):
            # self.scheduler.step()
            # self.lr = self.scheduler.get_lr()[0]

            search_result = self.train(self.dataset.train[0], self.dataset.train[1], self.args.batch_size)
            self.log_record(
                "=> train result: epoch{} - lr : {} - loss : {} - rse: {}  -rae : {}".format(i_epoch, self.lr,
                                                                                              search_result['loss'],
                                                                                              search_result['rse'],
                                                                                              search_result['rae']))
            # DecayScheduler().step(i_epoch)
            with torch.no_grad():
                val_result = self.infer(self.dataset.valid[0], self.dataset.valid[1], self.args.batch_size)
                self.log_record(
                    "=> valid result:{} - rse: {} - CORR : {}  -rae : {}".format(i_epoch, val_result['rse'],
                                                                                 val_result['correlation'],
                                                                                 val_result['rae']))

                test_result = self.infer(self.dataset.test[0], self.dataset.test[1], self.args.batch_size)
                self.log_record(
                    "=> test result:{} - rse: {} - CORR : {}  -rae : {}".format(i_epoch, test_result['rse'],
                                                                                test_result['correlation'],
                                                                                test_result['rae']))
            if float(val_result['rse']) < best_val:
                with open(self.args.save, 'wb') as f:
                    torch.save(self.model, f)
                best_val = float(val_result['rse'])
                b_epoch = i_epoch
                self.log_record(
                    "=> best valid epoch:{} - rse: {} ".format(b_epoch, best_val))

            if test_result['rse'] == "nan":
                self.log_record(
                    "=> break training at epoch:{}!!! ".format(b_epoch))
                break


        # Load the best saved model.
        with open(self.args.save, 'rb') as f:
            self.model = torch.load(f)

        val_result = self.infer(self.dataset.valid[0], self.dataset.valid[1], self.args.batch_size)

        self.log_record(
            "=> final valid result:{} - rse: {} - CORR : {}  -rae : {}".format(b_epoch, val_result['rse'],
                                                                         val_result['correlation'],
                                                                         val_result['rae']))

        test_result = self.infer(self.dataset.test[0], self.dataset.test[1], self.args.batch_size)

        self.log_record(
            "=> !!! final test result:{} - rse: {} - CORR : {}  -rae : {} !!!".format(b_epoch, test_result['rse'],
                                                                        test_result['correlation'],
                                                                        test_result['rae']))
        if test_result['rse'] == "nan":
            test_result['rse'] = str(1.0)
        with open(self.args.load_genotypes, "r") as f:
            genotypes      = yaml.load(f, Loader=yaml.FullLoader)
            genotypes['fitness'] = test_result['rse']
        with open(self.args.load_genotypes, "w") as f:
            yaml.dump(genotypes, f)


    


    def train(self, X, Y, batch_size):

        self.model.train()
        total_loss = 0
        eval_rse, eval_rae = 0, 0
        n_samples = 0
        iter = 0
        for tx, ty in self.dataset.get_batches(X, Y, batch_size, True):
            # self.model.zero_grad()
            # tx = torch.unsqueeze(tx, dim=1)
            # tx = tx.transpose(2, 3)

            # 模型训练
            self.optimizer.optimizer.zero_grad()
            # self.model.zero_grad()
            output = self.model(tx)
            output = torch.squeeze(output)
            scale = self.dataset.scale.expand(output.size(0), self.dataset.m)
            loss = self.loss_fn(output * scale, ty * scale)
            eval_rse += self.evaluateL2(output * scale, ty * scale).item()
            eval_rae += self.evaluateL1(output * scale, ty * scale).item()
            n_samples += (output.size(0) * self.dataset.m)
            rse = math.sqrt(eval_rse / n_samples) / self.dataset.rse
            rae = (eval_rae / n_samples) / self.dataset.rae
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * self.dataset.m)
            grad_norm = self.optimizer.step()

            iter += 1
        # return total_loss / n_samples
        return {'loss': total_loss / (iter + 1),
                    "rse": '{0:1.5f}'.format(rse), "rae": '{0:1.5f}'.format(rae)}



    def infer(self, X, Y, batch_size):

        self.model.eval()
        predict = None
        total_loss = 0
        eval_rse, eval_rae = 0, 0
        n_samples = 0
        iter = 0

        for tx, ty in self.dataset.get_batches(X, Y, self.args.batch, False):
            self.model.zero_grad()
            # tx = torch.unsqueeze(tx, dim=1)
            # tx = tx.transpose(2, 3)

            with torch.no_grad():
                output = self.model(tx)
            output = torch.squeeze(output)
            if len(output.shape) == 1:
                output = output.unsqueeze(dim=0)
            if predict is None:
                predict = output
                test = ty
            else:
                predict = torch.cat((predict, output))
                test = torch.cat((test, ty))

            scale = self.dataset.scale.expand(output.size(0), self.dataset.m)
            eval_rse += self.evaluateL2(output * scale, ty * scale).item()
            eval_rae += self.evaluateL1(output * scale, ty * scale).item()
            n_samples += (output.size(0) * self.dataset.m)



        rse = math.sqrt(eval_rse / n_samples) / self.dataset.rse
        rae = (eval_rae / n_samples) / self.dataset.rae

        predict = predict.data.cpu().numpy()
        Ytest = test.data.cpu().numpy()
        sigma_p = (predict).std(axis=0)
        sigma_g = (Ytest).std(axis=0)
        mean_p = predict.mean(axis=0)
        mean_g = Ytest.mean(axis=0)
        index = (sigma_g != 0)
        correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        correlation = (correlation[index]).mean()

        return {"rse": '{0:1.5f}'.format(rse), "rae": '{0:1.5f}'.format(rae),
                "correlation": '{0:1.5f}'.format(correlation)}


    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt.strftime( '%Y-%m-%d %H:%M:%S' )
        if first_time:
            file_mode = 'w'
        else:
            file_mode = 'a+'
        # f = open('./log/%s.txt'%(self.file_id), file_mode)
        f = open(self.args.log_save, file_mode)
        f.write('[%s]-%s\n'%(dt, _str))
        f.flush()
        f.close()



class RunModel(object):
    def do_work(self, args):
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device[-1])
        # device = 'cuda:0'
        start_time = time.time()
        t = Trainer(args)
        t.log_record("start training!", first_time=True)
        t.run()
        end_time = time.time()
        t.log_record("!!!!!!!!!!!!!!!!!!!!!!time: {:5.2f}s !!!!!!!!!!!!!!!!!!!!!!".format(end_time - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
    parser.add_argument('--data', type=str, default='./data/h5data/exchange-rate.h5',
                        help='location of the data file')
    parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='trained_model/model.pt',
                        help='path to save the final model')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--L1Loss', type=bool, default=True)
    parser.add_argument('--normalize', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:0', help='')
    parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
    parser.add_argument('--buildA_true', type=bool, default=True,
                        help='whether to construct adaptive adjacency matrix')
    parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
    parser.add_argument('--num_nodes', type=int, default=137, help='number of nodes/variables')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--subgraph_size', type=int, default=20, help='k')
    parser.add_argument('--node_dim', type=int, default=32, help='dim of nodes')
    parser.add_argument('--dilation_exponential', type=int, default=2, help='dilation exponential')
    parser.add_argument('--conv_channels', type=int, default=16, help='convolution channels')
    parser.add_argument('--residual_channels', type=int, default=16, help='residual channels')
    parser.add_argument('--skip_channels', type=int, default=32, help='skip channels')
    parser.add_argument('--end_channels', type=int, default=64, help='end channels')
    parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
    parser.add_argument('--seq_in_len', type=int, default=24 * 7, help='input sequence length')
    parser.add_argument('--seq_out_len', type=int, default=1, help='output sequence length')
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--layers', type=int, default=3, help='number of layers')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')

    parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay rate')

    parser.add_argument('--clip', type=int, default=5, help='clip')

    parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
    parser.add_argument('--tanhalpha', type=float, default=3, help='tanh alpha')

    # parser.add_argument('--epochs', type=int, default=1, help='')
    parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')
    parser.add_argument('--step_size', type=int, default=100, help='step_size')
    ###邻接矩阵
    parser.add_argument('--st_embedding_dim', type=int, default=40,
                        help='the dimension of static node representation')
    parser.add_argument('--fc_dim', type=int, default=72544, help='fc_dim')

    parser.add_argument('--steps_per_day', type=int, default=1)
    parser.add_argument('--add_time_in_day', type=bool, default=False)
    parser.add_argument('--add_day_in_week', type=bool, default=True)
    ###NAS
    # parser = argparse.ArgumentParser('Train_from_Genotype')
    parser.add_argument('--task', type=str, default='node_level')
    # parser.add_argument('--data',           type = str,             default = 'SBM_CLUSTER')
    parser.add_argument('--extra', type=str, default='')
    parser.add_argument('--in_dim_V', type=int, default=7)
    # parser.add_argument('--node_dim',       type = int,             default = 70)
    parser.add_argument('--nb_layers', type=int, default=4)
    parser.add_argument('--nb_nodes', type=int, default=4)
    parser.add_argument('--nb_classes', type=int, default=6)
    parser.add_argument('--leaky_slope', type=float, default=1e-2)
    parser.add_argument('--batchnorm_op', default=False, action='store_true')
    parser.add_argument('--nb_mlp_layer', type=int, default=4)
    # parser.add_argument('--dropout',        type = float,           default = 0.2)
    parser.add_argument('--pos_encode', type=int, default=0)

    parser.add_argument('--data_clip', type=float, default=1.0)
    parser.add_argument('--nb_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=64)
    # parser.add_argument('--lr',             type = float,           default = 1e-3)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lr_min', type=float, default=0.00005)
    parser.add_argument('--momentum', type=float, default=0.9)
    # parser.add_argument('--weight_decay',   type = float,           default = 0.0)
    parser.add_argument('--optimizer', type=str, default='ADAM')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--load_genotypes', type=str,
                        default="/home/zixuan/MTS_NAS/archs/test_1130_TC/data/h5data/exchange-rate/90.yaml")
    parser.add_argument('--load_TC_genotypes', type=str,
                        default="/home/zixuan/MTS_NAS/archs/test_1201_TC_FG/data/h5data/exchange-rate/TC_78.yaml")
    parser.add_argument('--report_freq', type=int, default=5)
    parser.add_argument('--arch_save', type=str, default='archs/train_test_TC')
    parser.add_argument('--GNN_depth_per_block', type=int, default=2)
    parser.add_argument('--expid', type=str, default='train_TC')

    args = parser.parse_args()


    # args.data = "./data/exchange_rate.txt"
    args.data = "./data/h5data/exchange-rate.h5"
    args.num_nodes = 8
    args.subgraph_size = 8
    args.batch_size = 4
    args.epochs = 100
    args.horizon = 3
    # args.device = 'cuda:0'
    args.lr = 0.0025
    args.lr = 0.0001
    # args.layers = 5
    args.lr_min = 0.0001

    args.epochs = 100
    # args.L1Loss = False

    args.load_genotypes = "/home/liangzixuan/ENAS_MTSF/archs/exchange_000.yaml"
    # args.save = "./trained_model/model-wind.pt" #test!!!!!!!!!
    # args.data = "./data/h5data/wind.h5"
    # args.num_nodes = 28
    # args.subgraph_size=20
    # args.batch_size = 128
    # args.epochs = 30
    args.GNN_depth_per_block = 1
    # args.horizon = 3
    # args.device = 'cuda:0'
    # args.lr = 0.0001
    # args.lr_min = 0.0001

    # args.data = "./data/h5data/solar-energy.h5"
    # args.num_nodes = 137
    # args.batch_size = 4
    # args.epochs = 30
    # # args.add_time_in_day = True
    # # args.steps_per_day = 6*24
    # args.lr = 0.0001
    #
    # args.data = "./data/h5data/electricity.h5"
    # args.num_nodes = 321
    # args.batch_size = 4
    # args.epochs = 30

    args.save = "./trained_model/" + args.data.split("/")[-1][:-3] + "/"
    args.log_save = "./logs/" + args.data.split("/")[-1][:-3] + "/"
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if not os.path.exists(args.log_save):
        os.makedirs(args.log_save)

    # genotypes_id_init

    args.save += genotypes_id + ".pt"
    args.log_save += genotypes_id + ".txt"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device[-1])
    args.device = 'cuda:0'
    RunModel().do_work(args)

"""

