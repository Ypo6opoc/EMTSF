import yaml
import argparse
import numpy as np
import copy
from evolve_utils import GPUTools, Log, Utils
import importlib
from multiprocessing import Process
import time, os, sys
import torch
from collections import defaultdict


parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--horizon', type=int, default=3)
parser.add_argument('--data',type=str,default='exchange-rate',help='data path')
parser.add_argument('--multi',type=bool,default=False,help='if multi-step forecasting')
args = parser.parse_args()

TC_OPS = ['TC_None', 'Conv 1x2', 'Conv 1x3', 'Conv 1x6', 'Conv 1x7']
GC_OPS = ['V_Min', 'V_Mean', 'V_Sum', 'V_Max']
GC_preOPS = ['V_None', 'V_I', 'V_Sparse', 'V_Dense', 'V_I', 'V_I', 'V_I', 'V_I']

args.firststage = 2
args.secondstage = 4



def crossover(origin_geno1, origin_geno2, crossprob, crosstimes):
    geno1, geno2 = copy.deepcopy(origin_geno1), copy.deepcopy(origin_geno2)
    GC_geno_len = len(geno1['GC_Genotype'])
    TC_geno_len = len(geno1['TC_Genotype'])
    position = np.random.choice(GC_geno_len + TC_geno_len, crosstimes, replace=False)
    # position = list(position)
    # for num, index in enumerate(position):

    for pos in position:
        # pos = int(pos)
        if pos >= GC_geno_len:
            curtype = 'TC_Genotype'
            pos -= GC_geno_len
        else:
            curtype = 'GC_Genotype'

        random_float = np.random.rand()
        if random_float < crossprob:
            tmp = copy.deepcopy(geno1[curtype][pos])
            geno1[curtype][pos] = copy.deepcopy(geno2[curtype][pos])
            geno2[curtype][pos] = copy.deepcopy(tmp)
    geno1['fitness'], geno2['fitness'] = 100, 100
    return geno1, geno2


def mutation(origin_geno, mutateprob, mutatetime):
    geno = copy.deepcopy(origin_geno)
    GC_geno_len = len(geno['GC_Genotype'])
    TC_geno_len = len(geno['TC_Genotype'])
    position = np.random.choice(GC_geno_len + TC_geno_len, mutatetime, replace=True)

    for pos in position:
        if pos >= GC_geno_len:
            curtype = 'TC_Genotype'
            pos -= GC_geno_len
        else:
            curtype = 'GC_Genotype'

        random_float = np.random.rand()
        if random_float < mutateprob:
            topo = geno[curtype][pos]
            topo_len = len(topo['topology'])
            topo_pos = np.random.choice(topo_len, 1, replace=True)[0]

            if curtype == 'GC_Genotype':
                topo_pos = np.random.choice(range(1, topo_len), 1, replace=True)[0]
            cur_dst, cur_src, cur_ops = geno[curtype][pos]['topology'][topo_pos]['dst'], \
                                        geno[curtype][pos]['topology'][topo_pos]['src'], \
                                        geno[curtype][pos]['topology'][topo_pos]['ops']

            candidate_src = list(range(cur_dst))
            # candidate_src = candidate_src.remove(cur_src)
            # new_src = np.random.choice(candidate_src, 1)[0]
            new_src = candidate_src[np.random.randint(len(candidate_src))]
            if curtype == 'GC_Genotype':
                if cur_dst>4:
                    limit = 3
                else:
                    limit = 1
                new_src = int(np.random.choice(range(limit, cur_dst), 1, replace=True)[0])
            else:
                if new_src != 0:
                    while new_src != 0 and geno[curtype][pos]['topology'][new_src-1]['ops'] == "TC_None":
                        new_src = candidate_src[np.random.randint(len(candidate_src))]

            if curtype == 'TC_Genotype':
                candidate_ops = TC_OPS
            elif cur_dst > args.firststage and cur_dst <= args.secondstage:
                candidate_ops = GC_OPS
            else:
                candidate_ops = GC_preOPS
            # candidate_ops = candidate_ops.remove(cur_ops)
            # new_ops = np.random.choice(candidate_ops, 1)[0]
            new_ops = candidate_ops[np.random.randint(len(candidate_ops))]
            if new_ops == 'TC_None':
                for item in geno[curtype][pos]['topology']:
                    if item['src'] == cur_dst:
                        candidate_ops = copy.deepcopy(TC_OPS)
                        candidate_ops.remove('TC_None')
                        new_ops = candidate_ops[np.random.randint(len(candidate_ops))]
            geno[curtype][pos]['topology'][topo_pos]['src'], geno[curtype][pos]['topology'][topo_pos][
                'ops'] = new_src, new_ops
    geno['fitness'] = 100
    return geno


def dump_geno(id, geno):
    name = args.data + '_'+'{:0>3d}'.format(id) + ".yaml"
    with open(args.spath + name, "w") as f:
        yaml.dump(geno, f)
    return name

def get_geno(name):
    with open(args.spath + name, "r") as f:
        genotypes = yaml.load(f, Loader=yaml.FullLoader)
    return genotypes

def init_population(max_p, template_geno, no_template=False):
    create_folder_if_not_exists(args.spath)
    # name = args.data + '{:0>3d}'.format(0) + ".yaml"
    # with open(args.spath + name, "w") as f:
    #     yaml.dump(template_geno, f)
    name = dump_geno(0, template_geno)
    populations = {}
    populations[name] = template_geno
    if no_template:
        start = 0
    else:
        start = 1
    for i in range(start, max_p):

        new_geno = mutation(template_geno, 1, 100)
        while new_geno in populations.values():
            new_geno = mutation(template_geno, 1, 100)
        # name = args.data + '{:0>3d}'.format(i) + ".yaml"
        # with open(args.spath + name, "w") as f:
        #     yaml.dump(new_geno, f)
        name = dump_geno(i, new_geno)
        populations[name] = new_geno
    return populations


def evolving(max_generation, template_geno, continue_generation=0):
    index = 0
    if continue_generation != 0:
        index += continue_generation * args.pops_in_generation
        generations = range(continue_generation, max_generation)
        # sorted_dict, rank_dict = update_rank(rank_dict=None, index=index, restart=True)
        if os.path.exists(args.population_pools):
            history_parents_names = torch.load(args.population_pools)
            parents_name = history_parents_names[-1]
        else:
            print("ERROR!!!!! no such file: population_pools!!!")
            return 0
        # parents = load_parents()
    else:
        generations = range(max_generation)
    for iteration in generations:
        print("current generation:{}".format(iteration))
        if iteration == 0:
            populations = init_population(args.pops_in_generation, template_geno, no_template=args.no_template)
            index += args.pops_in_generation
            rank_dict = None
            parents_name = []
            for item in populations.keys():
                parents_name.append(str(item))
        else:
            populations = {}
            # parents = {}
            offsprings = []
            # parent1, parent2 = get_parent_geno(sorted_dict)
            # bias = len(sorted_dict)


            for _ in range(args.crossover_times):
                parent1 = get_parent_geno(parents_name)
                parent2 = get_parent_geno(parents_name)
                while parent1['GC_Genotype'] == parent2['GC_Genotype'] and parent1['TC_Genotype'] == parent2['TC_Genotype']:
                    print("repeated parent!!!")
                    parent2 = get_parent_geno(parents_name)
                random_float = np.random.rand()
                if random_float < 0.9:
                    new_geno1, new_geno2 = crossover(parent1, parent2, crossprob=1, crosstimes=2)
                    offsprings.append(new_geno1)
                    offsprings.append(new_geno2)
                else:
                    offsprings.append(copy.deepcopy(parent1))
                    offsprings.append(copy.deepcopy(parent2))

            for _ in range(len(offsprings)):
                cur_geno = offsprings[_]
                random_float = np.random.rand()
                if random_float < 0.5:
                    # origin_geno = np.random.choice(list(populations.values())) ###check!
                    new_geno = mutation(cur_geno, mutateprob=1, mutatetime=6)
                    offsprings[_] = new_geno
                # while new_geno in populations.values():
                #     new_geno = mutation(origin_geno, mutateprob=0.2, mutatetime=6)

            for item in offsprings:
                name = dump_geno(index, item)
                populations[name] = item
                index += 1

        generate_to_python_file(populations)
        evaluate_offspring_linux(populations)

        # evaluate_offspring(populations)

        # sorted_dict, rank_dict = update_rank(rank_dict=rank_dict, index=index, restart=False)
        if iteration == 0:
            first_generation=True
        else:
            first_generation=False
        parents, parents_name  = update_population(parents_name, populations, first_generation)


def update_population(parents, offsprings, first_generation=False):

    rank_dict = {}
    new_parents = {}
    population_pool = []
    new_parents_name = []
    # for item in parents.keys():
    #     cur_genotypes = get_geno(item)
    #     rank_dict[item] = float(cur_genotypes['fitness'])
    #     rank_dict[item] = np.random.rand()       ####for test
    #     population_pool.append(item)
    for item in parents:
        cur_genotypes = get_geno(item)
        if float(cur_genotypes['fitness']) not in rank_dict.values():
            rank_dict[item] = float(cur_genotypes['fitness'])
        # rank_dict[item] = np.random.rand()       ####for test
        population_pool.append(item)
    for item in offsprings.keys():
        cur_genotypes = get_geno(item)
        if float(cur_genotypes['fitness']) not in rank_dict.values():
            rank_dict[item] = float(cur_genotypes['fitness'])
        # rank_dict[item] = np.random.rand()  ####for test
        population_pool.append(item)
    sorted_dict = sorted(rank_dict.items(), key=lambda x: x[1], reverse=False)
    for i in range(args.elites):
        file_name = str(sorted_dict[i][0])
        geno = get_geno(file_name)
        new_parents[file_name] = geno
        new_parents_name.append(file_name)
    for _ in range(args.pops_in_generation - args.elites):
        selected_item_1 = str(np.random.choice(population_pool))
        selected_item_2 = str(np.random.choice(population_pool))
        while selected_item_2 == selected_item_1:
            selected_item_2 = str(np.random.choice(population_pool))
        value1 = float(get_geno(selected_item_1)['fitness'])
        value2 = float(get_geno(selected_item_2)['fitness'])
        # value1 = np.random.rand()       ####for test
        # value2 = np.random.rand()       ####for test
        if value1 < value2:
            selected_item = selected_item_1
        else:
            selected_item = selected_item_2

        geno = get_geno(selected_item)
        new_parents[selected_item] = geno
        new_parents_name.append(selected_item)

    if first_generation:
        new_parents_names = []
        new_parents_names.append(parents)
        torch.save(new_parents_names, args.population_pools)
    else:
        if os.path.exists(args.population_pools):
            # with open("population_pools", "r") as f:
            #     new_parents_names = yaml.load(f, Loader=yaml.FullLoader)
            new_parents_names = torch.load(args.population_pools)
        else:
            new_parents_names = []
        new_parents_names.append(new_parents_name)
        torch.save(new_parents_names, args.population_pools)
    return new_parents, new_parents_name


# def get_parent_geno(sorted_dict):
#     name1, name2 = sorted_dict[0][0], sorted_dict[1][0]
#     geno1 = get_geno(name1)
#     geno2 = get_geno(name2)
#     print("Parent1:{} Fitness:{}".format(name1, geno1['fitness']))
#     print("Parent2:{} Fitness:{}".format(name2, geno2['fitness']))
#     return geno1, geno2

def get_parent_geno(parents_name):
    name1 = np.random.choice(parents_name)
    name2 = np.random.choice(parents_name)
    geno1 = get_geno(name1)
    geno2 = get_geno(name2)
    # geno1['fitness'] = np.random.rand()       ####for test
    # geno2['fitness'] = np.random.rand()       ####for test
    if float(geno1['fitness']) < float(geno2['fitness']):
        name, geno = name1, geno1
    else:
        name, geno = name2, geno2

    print("Parent:{} Fitness:{}".format(name, geno['fitness']))
    return geno


def generate_to_python_file(populations):
    Log.info('Begin to generate python files')

    for indi in populations:
        Utils.generate_pytorch_file(args.spath+indi, indi[:-5], args.pyfile_path, horizon=args.horizon, mulity=args.multi)
    Log.info('Finish the generation of python files')
    Log.info('Start training, you can use "nvidia-smi" to monitor the individual training processes running in the background. ')


def evaluate_offspring(populations):

    has_evaluated_offspring = False
    USE_PREDICTOR = False
    # populations = sorted(populations, reverse=True)
    for indi in populations:
        if USE_PREDICTOR == False:
            has_evaluated_offspring = True
            time.sleep(60)
            gpu_id = GPUTools.detect_available_gpu_id()
            while gpu_id is None:
                time.sleep(60)
                gpu_id = GPUTools.detect_available_gpu_id()
            if gpu_id is not None:
                file_name = indi[:-5]
                Log.info('Begin to train %s' % (file_name))
                module_name = 'scripts.%s' % (file_name)
                if module_name in sys.modules.keys():
                    Log.info('Module:%s has been loaded, delete it' % (module_name))
                    del sys.modules[module_name]
                    _module = importlib.import_module('.', module_name)
                else:
                    _module = importlib.import_module('.', module_name)
                _class = getattr(_module, 'RunModel')
                cls_obj = _class()
                p = Process(target=cls_obj.do_work, args=('%d' % (gpu_id), ))
                p.start()
        else:
            file_name = indi.id
            Log.info('%s has inherited the fitness as %.5f, no need to evaluate' % (file_name, indi.acc))
            f = open('./populations/after_%s.txt' % (file_name[4:6]), 'a+')
            f.write('%s=%.5f\n' % (file_name, indi.acc))
            f.flush()
            f.close()

    """
    once the last individual has been pushed into the gpu, the code above will finish.
    so, a while-loop need to be insert here to check whether all GPU are available.
    Only all available are available, we can call "the evaluation for all individuals
    in this generation" has been finished.

    """
    if has_evaluated_offspring:
        all_finished = False
        while all_finished is not True:
            time.sleep(60)
            all_finished = GPUTools.all_gpu_available()
    """
    the reason that using "has_evaluated_offspring" is that:
    If all individuals are evaluated, there is no needed to wait for 300 seconds indicated in line#47
    """
    """
    When the codes run to here, it means all the individuals in this generation have been evaluated, then to save to the list with the key and value
    Before doing so, individuals that have been evaluated in this run should retrieval their fitness first.
    """
    # if has_evaluated_offspring:
    #     file_name = './populations/after_%s.txt' % (self.individuals[0].id[4:6])
    #     assert os.path.exists(file_name) == True
    #     f = open(file_name, 'r')
    #     fitness_map = {}
    #     for line in f:
    #         if len(line.strip()) > 0:
    #             line = line.strip().split('=')
    #             fitness_map[line[0]] = float(line[1])
    #     f.close()
    #     for indi in self.individuals:
    #         if indi.acc == -1:
    #             if indi.id not in fitness_map:
    #                 self.log.warn(
    #                     'The individuals have been evaluated, but the records are not correct, the fitness of %s does not exist in %s, wait 120 seconds' % (
    #                     indi.id, file_name))
    #                 sleep(60)  #
    #             indi.acc = fitness_map[indi.id]
    # else:
    #     self.log.info('None offspring has been evaluated')

def get_current_avaliable_gpus(GPU_list):
    avaliable_GPUs = []
    with open("GPU_black_list", "r") as f:
        gpu_blacklist = yaml.load(f, Loader=yaml.FullLoader)

    for GPU in GPU_list[0]:
        if GPU not in gpu_blacklist:
            avaliable_GPUs.append(GPU)

    return avaliable_GPUs, gpu_blacklist


def evaluate_offspring_linux(populations):

    GPU_list = GPUTools._get_equipped_gpu_ids_and_used_gpu_info()
    current_tasks = []
    busy_GPUs = []
    avaliable_GPUs = []
    # avaliable_GPUs = []
    # gpu_blacklist = ['0']
    # for GPU in GPU_list[0]:
    #     if GPU not in gpu_blacklist:
    #         avaliable_GPUs.append(GPU)
        
    # for GPU in GPU_list[0]:
    #     avaliable_GPUs.append(GPU)

    for indi in populations:
        tmp_geno = get_geno(indi)
        if tmp_geno['fitness'] != 100:
            print("@@@Repeated@@@ geno:{} fitness:{}".format(indi, tmp_geno['fitness']))
            continue
        # while len(current_tasks) == len(GPU_list[0])*2:
        with open("GPU_black_list", "r") as f:
            gpu_blacklist = yaml.load(f, Loader=yaml.FullLoader)
        for GPU in GPU_list[0]:
            if GPU not in gpu_blacklist and GPU not in busy_GPUs and GPU not in avaliable_GPUs:
                avaliable_GPUs.append(GPU)
        while len(current_tasks) == len(GPU_list[0])-len(gpu_blacklist):
            time.sleep(300)
            for item in current_tasks:
                cur_geno = get_geno(item[0])
                if cur_geno['fitness'] != 100:
                    print("geno:{} fitness:{}".format(item[0], cur_geno['fitness']))
                    os.system("rm ./%s.py" % (item[0][:-5]))
                    current_tasks.remove(item)
                    avaliable_GPUs.append(item[1])
                    busy_GPUs.remove(item[1])

        current_GPU = avaliable_GPUs[0]
        avaliable_GPUs.remove(current_GPU)
        busy_GPUs.append(current_GPU)
        current_tasks.append((indi, current_GPU))

        os.system("cp ./scripts/%s.py ./%s.py" % (indi[:-5], indi[:-5]))
        # os.system("conda activate DGL")
        # my_exec = "nohup python -u %s.py --device 'cuda:%s' > record_GPU_%s.out 2>&1 &" % (
        # indi[:-5], current_GPU, current_GPU)
        #
        # result = subprocess.run([my_exec], stdout=subprocess.PIPE)
        # print(result.stdout.decode())
        python_path = "/home/liangzixuan/anaconda3/envs/DGL/bin/python"
        command = "nohup " + python_path + " -u %s.py --device 'cuda:%s' > record_GPU_%s.out 2>&1 &" % (
        indi[:-5], current_GPU, current_GPU)
        os.system(command)      ###这里一定要改回去！！！
        # print(current_tasks) #改回去

    while len(current_tasks) != 0:
        time.sleep(60)
        for item in current_tasks:
            cur_geno = get_geno(item[0])
            if cur_geno['fitness'] != 100:
                print("geno:{} fitness:{}".format(item[0], cur_geno['fitness']))
                os.system("rm ./%s.py" % (item[0][:-5]))
                current_tasks.remove(item)
                avaliable_GPUs.append(item[1])

    Log.info('Finish the evaluation of current populations')

        # os.system("conda activate DGL && nohup python -u %s.py --device 'cuda:%s' > record_GPU_%s.out 2>&1 &" % (indi[:-5], current_GPU, current_GPU))



def update_rank(rank_dict, index, restart=False):
    geno_names = os.listdir(args.spath)
    # if "train_test_TC" in geno_names:
    #     geno_names.remove("train_test_TC")
    if not rank_dict:
        rank_dict = {}
        # geno_names = sorted(geno_names, reverse=True)[:args.pops_in_generation]
    if restart:
        geno_names = sorted(geno_names, reverse=False)[:index]
    else:
        geno_names = sorted(geno_names, reverse=False)[index-args.pops_in_generation : index]

    for item in geno_names:
        # with open(args.spath+item, "r") as f:
        #     genotypes = yaml.load(f, Loader=yaml.FullLoader)
        cur_genotypes = get_geno(item)
        rank_dict[item] = float(cur_genotypes['fitness'])
        # rank_dict[item] = cur_genotypes['fitness'] + np.random.randint(10, 100)

    sorted_dict = sorted(rank_dict.items(), key=lambda x: x[1], reverse=False)

    return sorted_dict, rank_dict



def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)





if __name__ == "__main__":

    save_path = "./archs/template_arch.yaml"

    # args.data = "exchange-rate"
    # args.data = "METR-LA"
    #args.data = "PEMS-BAY"
    # args.data = "electricity"
    # args.horizon = 3
    # args.multi = False
    # args.multi = True


    args.spath = "./archs/" + args.data + "/"
    if not os.path.exists(args.spath):
        os.makedirs(args.spath)
    args.pyfile_path = "./scripts/"
    if not os.path.exists(args.pyfile_path):
        os.makedirs(args.pyfile_path)
    args.no_template = True


    args.GPU_blacklist = [ ]
    with open("GPU_black_list", "w") as f:
        yaml.dump(args.GPU_blacklist, f)

    args.pops_in_generation = 20
    args.crossover_times = 10
    args.elites = 4
    with open(save_path, "r") as f:
        genotypes = yaml.load(f, Loader=yaml.FullLoader)
    print("Origin Geno:\n", genotypes)

    args.population_pools = "./populations/" + args.data + "/"
    if not os.path.exists(args.population_pools):
        os.makedirs(args.population_pools)
    args.population_pools += "population_pools.pt"
    print(args.population_pools)

    evolving(20, genotypes, continue_generation=0)
   
