import numpy as np
import random
import math
import pickle
import argparse
from multiprocessing import Pool, cpu_count
import warnings
import logging
import time

class dict_sample:
	def __init__(self, M, s, K, N, epsi = 1/2, normalize = False, n_processes = 1, lowmem = False, thresh = 1/2, fixed_supp = [], n_subspaces = -1):
		self.M = M
		self.s = s
		self.K = K
		self.N = N
		# self.n_zeros = n_zeros
		self.fixed_supp = fixed_supp
		self.n_subspaces = self.N if n_subspaces == -1 else n_subspaces
		if n_processes == -1:
			max_avail_cpu = int(cpu_count())
			self.n_processes = max_avail_cpu
		else:
			self.n_processes = n_processes
		self.thresh = thresh
		self.lowmem = lowmem
		if N > 10**6 and not self.lowmem:
			warnings.warn("N is greater than 10^6 but lowmem mode not set. Setting lowmem automatically.")
			self.lowmem = True
		if self.n_processes == -1:
			self.n_processes = int(cpu_count())
		start = time.time()
		self.D = self.build_D()
		self.X = self.build_X()
		self.Y = np.dot(self.D,self.X)
		self.normflag = normalize
		if self.normflag:
			self.default_thresh = 1/(self.s**2)
			self.Y = self.Y/np.linalg.norm(self.Y, axis = 0)
		else:
			self.default_thresh = 1
		self.HSig_D = self.build_HSig_D()
		if not self.lowmem:
			self.corr = np.abs(np.dot(np.transpose(self.Y[:,:self.n_subspaces]),self.Y))
		else:
			self.uncorr_idx = self.get_corr_lowmem()

	def build_D(self):
		D = np.random.normal(0,1,(self.M,self.K))
		D = D/np.linalg.norm(D, axis = 0)
		return(D)

	def build_X(self):
		X = np.zeros((self.K,self.N))
		if self.n_processes == 1:
			for i in range(self.N):
				X[:,i] = self.get_Xcol(i)
		else:
			params = list(range(self.N))
			with Pool(processes=self.n_processes) as executor:
				Xcols = executor.map(self.get_Xcol, params)
			X = np.transpose(np.vstack(Xcols))
		return(X)
	
	def get_Xcol(self,i):
		Xcol = np.zeros(self.K)
		if i < len(self.fixed_supp):
			supp_elem = self.fixed_supp[i]
			remaining_rows = list(range(supp_elem))+list(range(supp_elem+1,self.K))
			rows = [supp_elem] + random.sample(remaining_rows,self.s-1)
			Xcol[rows] = 1 - 2*np.random.binomial(1,0.5,self.s)
		else:
			rows = random.sample(range(self.K),self.s)
			Xcol[rows] = 1 - 2*np.random.binomial(1,0.5,self.s)
		return Xcol
	
	# def get_Xcol(self,i):
	# 	Xcol = np.zeros(self.K)
	# 	if i < self.n_zeros:
	# 		start = 1+(self.s-1)*i
	# 		rows = [0]+list(range(start,start + self.s - 1))
	# 		Xcol[rows] = 1 - 2*np.random.binomial(1,0.5,self.s)
	# 	else:
	# 		rows = random.sample(range(self.K),self.s)
	# 		Xcol[rows] = 1 - 2*np.random.binomial(1,0.5,self.s)
	# 	return Xcol
	
	def build_HSig_D(self):
		HSig_D = np.zeros((self.M,self.M))
		for i in range(self.N):
			HSig_D = HSig_D + np.outer(self.Y[:,i],self.Y[:,i])
		return HSig_D/np.linalg.norm(HSig_D)
    
	def reload_DY(self):
		self.D = self.build_D()
		self.Y = np.dot(self.D,self.X)
		if self.normflag:
			self.Y = self.Y/np.linalg.norm(self.Y, axis = 0)
			
	def get_corr_lowmem(self):
		uncorr_idx = []
		if self.n_processes == 1:
			for i in range(self.N):
				uncorr_idx.append(self.get_uncorr_i(i))
		else:
			params = list(range(self.N))
			with Pool(processes=self.n_processes) as executor:
				uncorr_results = executor.map(self.get_uncorr_i, params)
			for i in range(self.N):
				uncorr_idx.append(uncorr_results[i])
		return(uncorr_idx)
	
	def get_uncorr_i(self, i):
		inners_i = np.dot(np.transpose(self.Y[:,i]),self.Y)
		uncorr_i = np.nonzero(np.abs(inners_i) < self.thresh)[0]
		return uncorr_i