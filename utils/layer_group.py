# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch import nn


def children(m):
    "Get children of `m`."
    return list(m.children())


def num_children(m: nn.Module) -> int:
    "Get number of children modules in `m`."
    return len(children(m))


class ParameterModule(nn.Module):
    "Register a lone parameter `p` in a module."

    def __init__(self, p: nn.Parameter):
        super().__init__()
        self.val = p

    def forward(self, x): return x


def children_and_parameters(m):
    "Return the children of `m` and its direct parameters not registered in modules."
    children = list(m.children())
    children_p = sum([[id(p) for p in c.parameters()] for c in m.children()], [])
    for p in m.parameters():
        if id(p) not in children_p: children.append(ParameterModule(p))
    return children


flatten_model = lambda m: sum(map(flatten_model, children_and_parameters(m)), []) if num_children(m) else [m]
