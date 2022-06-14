import numpy as np
import random
import numpy.linalg as la
import argparse
import dict_sample as ds
import time
from datetime import datetime
import logging
import pathlib
import csv
import single_subspace_recovery as ssr
from multiprocessing import Pool, cpu_count

class subspace_recovery:

	def __init__(self, DS, n_subspaces = -1, mode = 'quad_weight', thresh = 1/2):
		self.DS = DS
		self.thresh = thresh
		self.mode = mode
		self.n_subspaces = n_subspaces
		if self.n_subspaces == -1:
			self.n_subspaces = self.DS.N
		self.subspaces = []
		for i in range(self.n_subspaces):
			self.subspaces.append(self.recover_single_subspace(i))
		self.errs = []
		for ssr in self.subspaces:
			self.errs.append(ssr.err)

	def recover_single_subspace(self, i):
		return ssr.single_subspace_recovery(self.DS, i, mode = self.mode)