import torch
import numpy as np
import argparse
import time
from util import *
from trainer import Trainer
from net import gtnet
from net_new import train_net_multi
import yaml
from datetime import datetime
# os.environ['CUDA_VISIBLE_DEVICES']='3'
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()

parser.add_argument('--device',type=str,default='cuda:3',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')

parser.add_argument('--adj_data', type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=str_to_bool, default=True,help='whether to construct adaptive adjacency matrix')
parser.add_argument('--load_static_feature', type=str_to_bool, default=False,help='whether to load static feature')
parser.add_argument('--cl', type=str_to_bool, default=True,help='whether to do curriculum learning')

parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes/variables')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=20,help='k')
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=1,help='dilation exponential')

parser.add_argument('--conv_channels',type=int,default=32,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=32,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=64,help='skip channels')
parser.add_argument('--end_channels',type=int,default=128,help='end channels')


parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=12,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=12,help='output sequence length')

parser.add_argument('--layers',type=int,default=3,help='number of layers')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=5,help='clip')
parser.add_argument('--step_size1',type=int,default=2500,help='step_size')
parser.add_argument('--step_size2',type=int,default=100,help='step_size')


parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--seed',type=int,default=101,help='random seed')
parser.add_argument('--save',type=str,default='./save/',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='adj alpha')

parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')

parser.add_argument('--runs',type=int,default=10,help='number of runs')

#new
parser.add_argument('--GNN_depth_per_block', type=int, default=1)
parser.add_argument('--add_time_in_day', type=bool, default=False)
parser.add_argument('--add_day_in_week', type=bool, default=True)


args = parser.parse_args()
torch.set_num_threads(3)

args.data = "./data/METR-LA"
args.num_nodes = 207
args.adj_data = "./data/sensor_graph/adj_mx.pkl"
# args.load_genotypes = "/home/liangzixuan/ENAS_MTSF/MST_NAS/archs/template_arch.yaml"
args.load_genotypes = "/home/liangzixuan/ENAS_MTSF/MST_NAS/archs/template_arch_multi.yaml"
args.expid = 999
# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device[-1])

# args.epochs = 150
# args.step_size1 = 3750

# args.data = "./data/PEMS-BAY"
# args.num_nodes = 325
# args.adj_data = "./data/sensor_graph/adj_mx_bay.pkl"


args.save = "./trained_model/" + args.data.split("/")[-1] + "/"
args.log_save = "./logs/" + args.data.split("/")[-1] + "/"
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.exists(args.log_save):
    os.makedirs(args.log_save)

# genotypes_id_init

# args.load_genotypes = "/home/liangzixuan/ENAS_MTSF/MST_NAS/archs/exchange-rate_3/exchange-rate_319.yaml"
args.load_genotypes = "/home/liangzixuan/ENAS_MTSF/final_test/archs/template_arch_multi.yaml"
genotypes_id = "exchange-rate_319"

args.save += genotypes_id + ".pt"
args.log_save += genotypes_id + ".txt"
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device[-1])
args.device = 'cuda:0'
args.epochs = 10

def log_record(_str, first_time=None):
    dt = datetime.now()
    dt.strftime('%Y-%m-%d %H:%M:%S')
    if first_time:
        file_mode = 'w'
    else:
        file_mode = 'a+'
    # f = open('./log/%s.txt'%(self.file_id), file_mode)
    f = open(args.log_save, file_mode)
    f.write('[%s]-%s\n' % (dt, _str))
    f.flush()
    f.close()

def main(runid):
    # torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(args.seed)
    #load data
    device = torch.device(args.device)
    dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    predefined_A = load_adj(args.adj_data)
    predefined_A = torch.tensor(predefined_A)-torch.eye(args.num_nodes)
    predefined_A = predefined_A.to(device)

    # if args.load_static_feature:
    #     static_feat = load_node_feature('data/sensor_graph/location.csv')
    # else:
    #     static_feat = None

    if not os.path.isfile(args.load_genotypes):
        raise Exception('Genotype file not found!')
    else:
        with open(args.load_genotypes, "r") as f:
            genotypes = yaml.load(f, Loader=yaml.FullLoader)
            args.nb_layers = len(genotypes['GC_Genotype'])
            args.nb_nodes = len({edge['dst'] for edge in genotypes['GC_Genotype'][0]['topology']})
            args.tc_nodes = len({edge['dst'] for edge in genotypes['TC_Genotype'][0]['topology']})

    model = train_net_multi(args, genotypes['GC_Genotype'], genotypes['TC_Genotype'], args.gcn_true,
                                 args.buildA_true, args.gcn_depth, args.num_nodes,
                                 device, loss_fn=None, dropout=args.dropout, subgraph_size=args.subgraph_size,
                                 node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                                 conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                                 skip_channels=args.skip_channels, end_channels=args.end_channels,
                                 seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                                 layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha,
                                 layer_norm_affline=True, static_feat=None)


    print(args)
    log_record("{}".format(args))
    print('The recpetive field size is', model.receptive_field)
    log_record('The recpetive field size is{}'.format(model.receptive_field))
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)
    log_record('Number of model parameters is'.format(nParams))

    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len, scaler, device, args.cl)

    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    minl = 1e5
    for i in range(1,args.epochs+1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            # trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            if iter%args.step_size2==0:
                perm = np.random.permutation(range(args.num_nodes))
            num_sub = int(args.num_nodes/args.num_split)
            for j in range(args.num_split):
                # if j != args.num_split-1:
                #     id = perm[j * num_sub:(j + 1) * num_sub]
                # else:
                #     id = perm[j * num_sub:]
                # id = torch.tensor(id).to(device)
                # tx = trainx[:, :, id, :]
                # ty = trainy[:, :, id, :]
                # metrics = engine.train(tx, ty[:,0,:,:],id)

                tx = trainx
                ty = trainy
                metrics = engine.train(tx, ty[:,0,:,:],None)

                train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Task Level: {:.4f}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, engine.task_level, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
                log_record(log.format(iter, engine.task_level, train_loss[-1], train_mape[-1], train_rmse[-1]))

        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            # testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        log_record(log.format(i,(s2-s1)))

        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        log_record(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)))

        #test_data
        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = realy.transpose(1, 3)[:, 0, :, :]

        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            # testx = testx.transpose(1, 3)
            with torch.no_grad():
                preds = engine.model(testx)
                preds = preds.transpose(1, 3)
            outputs.append(preds.squeeze())

        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:realy.size(0), ...]

        mae = []
        mape = []
        rmse = []
        for i in range(args.seq_out_len):
            pred = scaler.inverse_transform(yhat[:, :, i])
            real = realy[:, :, i]
            metrics = metric(pred, real)
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            # print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
            mae.append(metrics[0])
            mape.append(metrics[1])
            rmse.append(metrics[2])

        mae = np.array(mae)
        mape = np.array(mape)
        rmse = np.array(rmse)

        print('test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean')
        log_record('test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean')
        for i in [2, 5, 11]:
            log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}'
            print(log.format(i + 1, mae[i], rmse[i], mape[i]))
            log_record(log.format(i + 1, mae[i], rmse[i], mape[i]))



        if mvalid_loss<minl:
            # if not os.path.exists(args.save):
            #     os.makedirs(args.save)
            # torch.save(engine.model.state_dict(), args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth")
            with open(args.save, 'wb') as f:
                torch.save(engine.model, f)
            minl = mvalid_loss
            # best_epoch = i

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    log_record("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    log_record("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        engine.model = torch.load(f)
    # bestid = np.argmin(his_loss)
    # engine.model.load_state_dict(torch.load(args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth"))

    print("Training finished")
    log_record("Training finished")
    print("Load the best model according to valid loss")
    log_record("Load the best model according to valid loss")
    # print("The valid loss on best model is", str(round(his_loss[bestid],4)))
    # print("The best model id is {}".format(bestid))

    #valid data
    outputs = []
    realy = torch.Tensor(dataloader['y_val']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        # testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = engine.model(testx)
            preds = preds.transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]


    pred = scaler.inverse_transform(yhat)
    vmae, vmape, vrmse = metric(pred,realy)

    #test data
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        # testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    mae = []
    mape = []
    rmse = []
    for i in range(args.seq_out_len):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        log_record(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        mae.append(metrics[0])
        mape.append(metrics[1])
        rmse.append(metrics[2])
    return vmae, vmape, vrmse, mae, mape, rmse

if __name__ == "__main__":
    log_record("start training!", first_time=True)

    vmae = []
    vmape = []
    vrmse = []
    mae = []
    mape = []
    rmse = []
    start_time = time.time()
    for i in range(1):
        vm1, vm2, vm3, m1, m2, m3 = main(i)
        vmae.append(vm1)
        vmape.append(vm2)
        vrmse.append(vm3)
        mae.append(m1)
        mape.append(m2)
        rmse.append(m3)
    end_time = time.time()
    print("!!!!!!!!!!!!!!!!!!!!!!time: {:5.2f}s !!!!!!!!!!!!!!!!!!!!!!".format(end_time - start_time))
    log_record("!!!!!!!!!!!!!!!!!!!!!!time: {:5.2f}s !!!!!!!!!!!!!!!!!!!!!!".format(end_time - start_time))
    mae = np.array(mae)
    mape = np.array(mape)
    rmse = np.array(rmse)

    amae = np.mean(mae,0)
    amape = np.mean(mape,0)
    armse = np.mean(rmse,0)

    smae = np.std(mae,0)
    smape = np.std(mape,0)
    srmse = np.std(rmse,0)

    print('\n\nResults for 10 runs\n\n')
    log_record('\n\nResults for 10 runs\n\n')
    #valid data
    print('valid\tMAE\tRMSE\tMAPE')
    log_record('valid\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(vmae),np.mean(vrmse),np.mean(vmape)))
    log_record(log.format(np.mean(vmae),np.mean(vrmse),np.mean(vmape)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(vmae),np.std(vrmse),np.std(vmape)))
    log_record(log.format(np.std(vmae),np.std(vrmse),np.std(vmape)))
    print('\n\n')
    log_record('\n\n')
    #test data
    print('test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean\tMAE-std\tRMSE-std\tMAPE-std')
    log_record('test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean\tMAE-std\tRMSE-std\tMAPE-std')
    for i in [2,5,11]:
        log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
        print(log.format(i+1, amae[i], armse[i], amape[i], smae[i], srmse[i], smape[i]))
        log_record(log.format(i+1, amae[i], armse[i], amape[i], smae[i], srmse[i], smape[i]))





