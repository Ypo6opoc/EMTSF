import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs): 
    return torch.cat([x.view(-1) for x in xs])


class Architect(object): 

    def __init__(self, model, args): 
        self.args                 = args
        self.network_momentum     = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model                = model
        self.optimizer            = torch.optim.Adam(
            params       = self.model.arch_parameters(),
            lr           = args.arch_lr,
            betas        = (0.5, 0.999),
            weight_decay = args.arch_weight_decay
        )

    def _compute_unrolled_model(self, input, target, eta, network_optimizer): 
        loss  = self.model._loss(input, target)
        theta = _concat(self.model.parameters()).data
        try: 
            moment = _concat(network_optimizer.state[v]['momentum_buffer']
                             for v in self.model.parameters()).mul_(self.network_momentum)
        except: 
            moment = torch.zeros_like(theta)
            dtheta = _concat(torch.autograd.grad(
                            outputs      = loss,
                            inputs       = self.model.parameters())
                        ).data + self.network_weight_decay*theta
            unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled, dataset):
        self.optimizer.zero_grad()
        if unrolled: 
            self._backward_step_unrolled(
                input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else: 
            if self.args.search_mode == 'darts_1':
                self._backward_step(input_valid, target_valid, dataset)
            elif self.args.search_mode == 'train':
                self._backward_step(input_train, target_train, dataset)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid, dataset):
        loss = self.model._loss(input_valid, target_valid, dataset)
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer): 
        unrolled_model = self._compute_unrolled_model(
            input_train, target_train, eta, network_optimizer)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        unrolled_loss.backward()
        dalpha         = [v.grad for v in unrolled_model.arch_parameters()]
        vector         = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads): 
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha): 
            if  v.grad is None: 
                v.grad = Variable(g.data)
            else: 
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta): 
        model_new  = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters(): 
            v_length  = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset   += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        if not model_new.args.disable_cuda: 
            model_new.cuda = model_new.cuda()
        return model_new

    def _hessian_vector_product(self, vector, input, target, r=1e-2): 
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector): 
            p.data.add_(R, v)
        loss    = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector): 
            p.data.sub_(2*R, v)
        loss    = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector): 
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

class TC_Architect(object):

    def __init__(self, model, args):
        self.args                 = args
        self.network_momentum     = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model                = model
        self.optimizer            = torch.optim.Adam(
            params       = self.model.TC_arch_parameters(),
            lr           = args.arch_lr,
            betas        = (0.5, 0.999),
            weight_decay = args.arch_weight_decay
        )

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss  = self.model._loss(input, target)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer']
                             for v in self.model.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
            dtheta = _concat(torch.autograd.grad(
                            outputs      = loss,
                            inputs       = self.model.parameters())
                        ).data + self.network_weight_decay*theta
            unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled, dataset):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(
                input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            if self.args.search_mode == 'darts_1':
                self._backward_step(input_valid, target_valid, dataset)
            elif self.args.search_mode == 'train':
                self._backward_step(input_train, target_train, dataset)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid, dataset):
        loss = self.model._loss(input_valid, target_valid, dataset)
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(
            input_train, target_train, eta, network_optimizer)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        unrolled_loss.backward()
        dalpha         = [v.grad for v in unrolled_model.arch_parameters()]
        vector         = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if  v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new  = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length  = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset   += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        if not model_new.args.disable_cuda:
            model_new.cuda = model_new.cuda()
        return model_new

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss    = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2*R, v)
        loss    = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]



class All_Architect(object):

    def __init__(self, model, args):
        self.args                 = args
        self.network_momentum     = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model                = model
        self.optimizer_TC            = torch.optim.Adam(
            params       = self.model.TC_arch_parameters(),
            lr           = args.arch_lr,
            betas        = (0.5, 0.999),
            weight_decay = args.arch_weight_decay
        )
        self.optimizer_GC = torch.optim.Adam(
            params=self.model.arch_parameters(),
            lr=args.arch_lr,
            betas=(0.5, 0.999),
            weight_decay=args.arch_weight_decay
        )
        self.optimizer_ALL = torch.optim.Adam(
            params=self.model.arch_parameters()+self.model.TC_arch_parameters(),
            lr=args.arch_lr,
            betas=(0.5, 0.999),
            weight_decay=args.arch_weight_decay
        )


    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled, dataset, optimize_mode=None):
        if optimize_mode:
            if optimize_mode == "TC":
                optimizer = self.optimizer_TC
            elif optimize_mode == "GC":
                optimizer = self.optimizer_GC
            elif optimize_mode == "ALL":
                optimizer = self.optimizer_ALL
        else:
            print("optimize_mode error!")
        optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(
                input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            if self.args.search_mode == 'darts_1':
                self._backward_step(input_valid, target_valid, dataset)
            elif self.args.search_mode == 'train':
                self._backward_step(input_train, target_train, dataset)
        optimizer.step()

    def _backward_step(self, input_valid, target_valid, dataset):
        loss = self.model._loss(input_valid, target_valid, dataset)
        loss.backward()







