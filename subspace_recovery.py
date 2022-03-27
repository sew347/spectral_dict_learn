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

	def __init__(self, DS, thresh, num_subspaces = -1, parallel = False, n_processes = None):
		self.DS = DS
		self.thresh = thresh
		self.num_subspaces = num_subspaces
		if self.num_subspaces == -1:
			self.num_subspaces = self.DS.N
		if not parallel:
			self.subspaces = []
			for i in range(self.num_subspaces):
				self.subspaces.append(self.recover_single_subspace(i))
		else:
			if n_processes is None:
				max_avail_cpu = int(cpu_count()*(2/3))
				if self.num_subspaces < max_avail_cpu:
					n_processes = self.num_subspaces
				else:
					n_processes = max_avail_cpu
			params = list(range(self.num_subspaces))
			with Pool(processes=n_processes) as executor:
				self.subspaces = executor.map(self.recover_single_subspace, params)

	def recover_single_subspace(self, i):
		return ssr.single_subspace_recovery(self.DS, self.thresh, i)