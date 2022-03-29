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
				self.intersections.append(self.get_intersections_i(i))
		else:
			max_avail_cpu = int(cpu_count())
			if n_processes > max_avail_cpu or n_processes == -1:
				n_processes = max_avail_cpu
			if (self.max_idx-1) < n_processes:
				n_processes = self.max_idx - 1
			else:
				n_processes = max_avail_cpu
			params = set(range(self.max_idx-1))
			with Pool(processes=n_processes) as executor:
				self.intersections = executor.map(self.get_intersections_i, params)
		
	def get_intersections_i(self, i):
		SI_i = []
		for j in range(i+1,self.max_idx):
			SSI = ssi.single_subspace_intersection(self.SR,i,j,delta = self.delta)
			if SSI.emp_uniq_int_flag:
				if self.is_new_dhat(SI_i,SSI.dhat):
					SI_i.append(SSI)
			if len(SI_i) >= self.SR.DS.s:
				break
		return SI_i

	def is_new_dhat(self,SI_i,dhat):
		for SSI in SI_i:
			if np.abs(np.inner(SSI.dhat,dhat)) > self.delta:
				return False
		return True
