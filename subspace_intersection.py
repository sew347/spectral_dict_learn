import numpy as np
import random
import numpy.linalg as la
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
from multiprocessing import Pool, cpu_count
from itertools import combinations

class subspace_intersection:
	def __init__(self, SR, delta = 0.5, max_idx = -1, parallel = False, n_processes = None):
		self.SR = SR
		self.delta = delta
		self.max_idx = max_idx
		if self.max_idx == -1 or self.max_idx > self.SR.n_subspaces:
			self.max_idx = self.SR.n_subspaces
		self.intersections = []
		if not parallel or n_processes == 1:
			for i in range(self.max_idx-1):
				for j in range(i+1, self.max_idx):
					self.intersections.append(self.get_single_intersection([i,j]))
		else:
			max_avail_cpu = int(cpu_count())
			n_iter = int(self.max_idx*(self.max_idx-1)/2)
			if n_processes > max_avail_cpu or n_processes == -1:
				n_processes = max_avail_cpu
			if self.n_subspaces < n_processes:
				n_processes = self.n_subspaces
			else:
				n_processes = max_avail_cpu
			elems = set(range(self.max_idx))
			params = list(combinations(set(elems), 2))
			with Pool(processes=n_processes) as executor:
				self.intersections = executor.map(self.get_single_intersection, params)
		
	def get_single_intersection(self, idx):
		SI = ssi.single_subspace_intersection(self.SR,idx[0],idx[1], delta = self.delta)
		return SI
	

	# def get_true_intersections(self):
	# 	intersect_matrix = np.dot(np.transpose(np.abs(self.SR.DS.X[:,:self.max_idx])),np.abs(self.SR.DS.X[:,:self.max_idx]))
	# 	np.fill_diagonal(intersect_matrix, 0)
	# 	intersect_matrix *= 1 - np.tri(*intersect_matrix.shape, k=-1)
	# 	all_true_int = np.nonzero(intersect_matrix)
	# 	print(np.shape(all_true_int)[0])
	# 	true_uniq_int = {}
	# 	for m in range(np.shape(all_true_int[0])[0]):
	# 		i = all_true_int[0][m]
	# 		j = all_true_int[1][m]
	# 		supp_i = np.nonzero(self.SR.DS.X[:,i])
	# 		supp_j = np.nonzero(self.SR.DS.X[:,j])
	# 		intersection = np.intersect1d(supp_i,supp_j)
	# 		if np.shape(intersection)[0] == 1:
	# 			true_uniq_int[(i,j)] =intersection
	# 	return true_uniq_int