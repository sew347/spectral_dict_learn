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
		self.cluster = self.single_o_cluster() if not self.CR.complement_mode else self.single_o_cluster_complement()
		self.dhat = self.recover_single_element()
		self.true_d, self.true_uniq_int_flag, self.true_idx = self.true_subspace_intersection()
		if self.true_uniq_int_flag:
			self.inner = np.abs(np.inner(self.dhat, self.true_d))
		else:
			self.inner = -1

	def single_o_cluster(self):
		cluster = [self.i,self.j]
		nbhd = np.intersect1d(list(self.CR.G.neighbors(self.i)),list(self.CR.G.neighbors(self.j)))
		for n in self.CR.G.nodes:
			if n != self.i and n != self.j:
				int_nbhd = np.intersect1d(nbhd, list(self.CR.G.neighbors(n)))
				if len(int_nbhd) > self.T:
					cluster.append(n)
		return cluster
	
	def single_o_cluster_complement(self):
		cluster = [self.i,self.j]
		non_nbhd = np.union1d(list(self.CR.G.neighbors(self.i)),list(self.CR.G.neighbors(self.j)))
		for n in self.CR.G.nodes:
			if n != self.i and n != self.j:
				union_non_nbhd = np.union1d(non_nbhd, list(self.CR.G.neighbors(n)))
				if (self.CR.DS.N-len(union_non_nbhd)) > self.T:
					cluster.append(n)
		return cluster
		
	def recover_single_element(self):
		sig_cluster = np.zeros((self.CR.DS.M,self.CR.DS.M))
		for c in self.cluster:
			sig_cluster = sig_cluster + np.outer(self.CR.DS.Y[:,c],self.CR.DS.Y[:,c])
		sig_cluster_proj = self.build_sig_cluster_proj(sig_cluster)
		E = la.eigh(sig_cluster_proj)
		dhat = E[1][:,-1]
		return dhat
	
	def build_sig_cluster_proj(self,sig_cluster):
		inner_prod = np.trace(np.dot(sig_cluster,self.CR.DS.HSig_D))
		return sig_cluster - inner_prod*self.CR.DS.HSig_D
	
	def true_subspace_intersection(self):
		supp_i = np.nonzero(self.CR.DS.X[:,self.i])
		supp_j = np.nonzero(self.CR.DS.X[:,self.j])
		supp_int = np.intersect1d(supp_i,supp_j)
		if np.shape(supp_int)[0]==1:
			return (self.CR.DS.D[:,supp_int[0]], True, supp_int[0])
		else:
			return (0, False, -1/2)
