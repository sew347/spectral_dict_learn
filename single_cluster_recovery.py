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
import single_subspace_recovery as ssr
import single_subspace_intersection as ssi
from itertools import combinations
import networkx as nx

class single_cluster_recovery:
	def __init__(self, CR, i, j):
		self.CR = CR
		self.i = i
		self.j = j
		self.T = self.CR.DS.N*self.CR.DS.s/10/self.CR.DS.K
		self.cluster = self.single_o_cluster()
		self.dhat = self.recover_single_element()
		self.true_idx = int(i/2)
		self.true_d = self.CR.DS.D[:,self.true_idx]
		self.inner = np.abs(np.inner(self.dhat, self.true_d))

	def single_o_cluster(self):
		cluster = [self.i,self.j]
		nbhd = np.intersect1d(list(self.CR.G.neighbors(self.i)),list(self.CR.G.neighbors(self.j)))
		for n in self.CR.G.nodes:
			if n != self.i and n != self.j:
				int_nbhd = np.intersect1d(nbhd, list(self.CR.G.neighbors(n)))
				if len(int_nbhd) > self.T:
					cluster.append(n)
		return cluster
		
	def recover_single_element(self):
		sig_cluster = np.zeros((self.CR.DS.M,self.CR.DS.M))
		for c in self.cluster:
			sig_cluster = sig_cluster + np.outer(self.CR.DS.Y[:,c],self.CR.DS.Y[:,c])
		E = la.eigh(sig_cluster)
		dhat = E[1][:,-1]
		return dhat
	
