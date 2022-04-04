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

class subspace_recovery:

	def __init__(self, DS, thresh, n_subspaces = -1, parallel = False, n_processes = 1):
		self.DS = DS
		self.thresh = thresh
		self.n_subspaces = n_subspaces
		if self.n_subspaces == -1:
			self.n_subspaces = self.DS.N
		if not parallel or n_processes == 1:
			self.subspaces = []
			for i in range(self.n_subspaces):
				self.subspaces.append(self.recover_single_subspace(i))
		else:
			max_avail_cpu = int(cpu_count())
			if n_processes > max_avail_cpu or n_processes == -1:
				n_processes = max_avail_cpu
			if self.n_subspaces < n_processes:
				n_processes = self.n_subspaces
			else:
				n_processes = max_avail_cpu
			params = list(range(self.n_subspaces))
			with Pool(processes=n_processes) as executor:
				self.subspaces = executor.map(self.recover_single_subspace, params)
		self.errs = []
		for ssr in self.subspaces:
			self.errs.append(ssr.err)

	def recover_single_subspace(self, i):
		return ssr.single_subspace_recovery(self.DS, self.thresh, i)