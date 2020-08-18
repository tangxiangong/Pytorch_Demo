#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/8/17
# @Author  : Tang Xiangong
# @Contact : tangxg16@lzu.edu.cn
# @File    : pruner.py
import torch
import types


class Pruner(object):
    def __init__(self, model, module_type, prune_rate=0.0, regularizer=None):
        self.model = model
        self.module_type = module_type
        self.prune_rate = prune_rate
        self.regularizer = regularizer
        self.gates_param = []
        self.masks = []

    def init_gates(self):
        for m in self.model.modules():
            if m.__class__.__name__ == self.module_type:
                m.init_gates()

    def collect_gates(self):
        for m in self.model.modules():
            if m.__class__.__name__ == self.module_type:
                self.gates_param.extend(m.get_gates())

    def replace_forward(self):
        for m in self.model.modules():
            if m.__class__.__name__ == self.module_type:
                m.forward = types.MethodType(m.gated_forward, m)

    def gates_loss(self):
        return self.regularizer(self.gates_param)

    def calculate_mask(self):
        gates_lens = [len(gate) for gate in self.gates_param]
        all_gates = torch.cat(self.gates_param).abs()
        keep_gate_num = int(all_gates.numel()*(1-self.prune_rate))
        keep_idx = all_gates.topk(keep_gate_num, 0)[1]
        masks = torch.zeros_like(all_gates)
        masks[keep_idx] = 1.0
        self.masks = torch.split_with_sizes(masks, gates_lens)

    def zerout_gates(self):
        for i in range(len(self.gates_param)):
            self.gates_param[i].data.mul_(self.masks[i])

    def export_pruned_cfg(self):
        cfg = []
        for m in self.masks:
            keep_num = max(m.sum().long().item(), 1)
            cfg.append(keep_num)
        return cfg


class LnRegularizer(object):
    def __init__(self, order=1):
        self.order = order

    def __call__(self, gates):
        all_gates = torch.cat(gates)
        ln_norm = torch.norm(all_gates, p=self.order)/len(all_gates)
        return ln_norm


def main():
    pass


if __name__ == "__main__":
    main()