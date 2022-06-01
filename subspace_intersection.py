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
	def __init__(self, SR, tau = 0.5, max_idx = -1, n_processes = 1):
		self.SR = SR
		self.tau = tau
		self.max_idx = max_idx
		if self.max_idx == -1 or self.max_idx > self.SR.n_subspaces:
			self.max_idx = self.SR.n_subspaces
		self.intersections = []
		self.recovered = []
		for i in range(self.max_idx-1):
			SI_i,R_i = self.get_intersections_i(i)
			self.intersections.append(SI_i)
			self.recovered.append(R_i)

	def get_intersections_i(self, i):
		#memory is typically too limited to hold all intersections; only store cases with empirical or true intersection
		#SI_i: any intersection with a true or empirical unique intersection
		SI_i = []
		#R_i: only recovered dict elements
		R_i = []
		for j in range(i+1,self.max_idx):
			SSI = ssi.single_subspace_intersection(self.SR,i,j,tau = self.tau)
			if SSI.emp_uniq_int_flag or SSI.true_uniq_int_flag:
				SI_i.append(SSI)
				if SSI.emp_uniq_int_flag:
					if self.is_new_dhat(R_i,SSI.dhat):
						R_i.append(SSI)
				if len(R_i) >= self.SR.DS.s:
					break
		return SI_i, R_i

	def is_new_dhat(self,SI_i,dhat):
		for SSI in SI_i:
			if np.abs(np.inner(SSI.dhat,dhat)) > self.tau:
				return False
		return True
