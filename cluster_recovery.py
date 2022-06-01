import numpy as np
import random
import numpy.linalg as la
import math
import argparse
import dict_sample as ds
import matplotlib.pyplot as plt
import time
from datetime import datetime
import logging
import pathlib
import csv
import single_cluster_recovery as scr
import warnings
from itertools import combinations
import networkx as nx

class cluster_recovery:
	def __init__(self, DS, thresh = 0.5, lowmem = False, block_size = 500, max_idx = -1, complement_mode = False):
		self.DS = DS
		self.T = DS.N*DS.s/10/DS.K
		self.thresh = 0.5
		self.lowmem = lowmem
		self.block_size = block_size
		self.complement_mode = complement_mode
		if self.complement_mode:
			logging.info('Complement mode is set to True.')
		if not self.lowmem and self.DS.N > 40000:
			warnings.warn('Lowmem not set for N > 40000. Setting automatically.')
			self.lowmem = True
		self.max_idx = max_idx
		if self.max_idx == -1:
			self.max_idx = DS.n_subspaces
		self.G = self.build_graph()
		self.recovered = self.recover_dict_elements()
		
	def build_graph(self):
		if not self.lowmem:
			logging.info('Generating graph in standard mode.')
			corr = np.abs(np.dot(np.transpose(self.DS.Y),self.DS.Y))
			logging.info('Correlation data generated.')
			adjacency = (corr>self.thresh) if not self.complement_mode else (corr<=self.thresh)
			if not self.complement_mode:
				np.fill_diagonal(adjacency,0)
			else:
				np.fill_diagonal(adjacency,1)
			logging.info('Generating graph from adjacency.')
			G = nx.from_numpy_array(adjacency)
			logging.info('Graph generated.')
			return G
		else:
			logging.info('Generating graph in lowmem mode.')
			n_blocks = math.ceil(self.DS.N/self.block_size)
			G = nx.empty_graph(self.DS.N)
			for i in range(n_blocks):
				idx = i*self.block_size
				corr_block = np.abs(np.dot(np.transpose(self.DS.Y[:,idx:(idx+self.block_size)]),self.DS.Y))
				adj_block = list(np.nonzero(corr_block > self.thresh)) if not self.complement_mode else list(np.nonzero(corr_block <= self.thresh))
				adj_block[0] = adj_block[0]+idx
				edges = list(map(tuple, np.transpose(adj_block)))
				G.add_edges_from(edges)
			diag_indices = [(i,i) for i in range(self.DS.N)]
			if not self.complement_mode:
				G.remove_edges_from(diag_indices)
			else:
				G.add_edges_from(diag_indices)
			return G
	
	#recover dictionary elements by clustering
	#specialized to test protocol in cluster_test_script; not for general use
	def recover_dict_elements(self):
		recovered = []
		for i in range(math.floor(self.max_idx/2)):
			SCR = scr.single_cluster_recovery(self, 2*i,2*i+1)
			recovered.append(SCR)
		return recovered