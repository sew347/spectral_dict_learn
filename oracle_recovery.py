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
import single_oracle_recovery as sor
from multiprocessing import Pool, cpu_count

# oracle_recovery.py
#### summary ####
#runs recovery using a specified oracle
#### inputs ####
#DS = dict_sample (data)
#n_cols = number of dictionary elements to recover. If set, true dictionary elements will be used as oracles.
#SI = subspace_intersection object. If set, dictionary elements will be the estimated elements from SI.
#*ONLY ONE* of n_cols, SI should be set at a time.
#### fields ####
#DS = input DS
#cols = recovered dictionary elements
#inners = array of inner products of recovered with true dictionary elememts

class oracle_recovery:

	def __init__(self, DS, n_cols = -1, SI = None):
		if n_cols == -1 and SI is None:
			raise ValueError('Either n_cols or SI must be provided.')
		if n_cols != -1 and SI is not None:
			raise ValueError('Only one of n_cols and SI should be provided.')
		self.DS = DS
		self.cols = []
		#if n_cols provided, use first n_cols dictionary elements as oracle and truth
		if n_cols != -1:
			self.n_cols = n_cols
			for k in range(self.n_cols):
				self.cols.append(sor.single_oracle_recovery(DS, DS.D[:,k], DS.D[:,k]))
		#if SI provided, use recovered SI as oracles and corresponding dict elements as truth
		else:
			for SI_i in SI.intersections:
				for SSI in SI_i:
					if SSI.true_uniq_int_flag and SSI.emp_uniq_int_flag:
						self.cols.append(sor.single_oracle_recovery(DS, SSI.dhat, SSI.d))
		self.inners = []
		for SOR in self.cols:
			self.inners.append(SOR.inner)