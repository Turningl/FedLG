# -*- coding: utf-8 -*-
# @Author : liang
# @File : __init__.py


from fedlg.graph.gat import GAT
from fedlg.graph.gcn import GCN
from fedlg.graph.mpnn import MPNN
from fedlg.graph.pool import GlobalPool, dot_and_global_pool
from fedlg.graph.s2s import Set2Set
from fedlg.graph.norm import Linear_BatchNorm1d