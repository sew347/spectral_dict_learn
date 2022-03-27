import numpy as np
import random
import math
import pickle
import argparse
#import scipy.sparse.linalg as spla
#from sklearn.preprocessing import normalize
from scipy.sparse import csc_matrix
from multiprocessing import Pool, cpu_count


#iterative refinement of subspace estimate

class dict_sample:
	def __init__(self, M, s, K, N, verbose = None, epsi = 1/2, normalize = False, n_zeros = 2):
		if M is None and s is None:
			raise AttributeError("Either M or s must not be None.")
		self.M = M if M is not None else math.ceil((s/epsi)**(0.5))
		self.s = s if s is not None else math.ceil(self.M*epsi)
		self.K = K if K is not None else math.ceil(1*(self.M**1))
		self.N = N if N is not None else math.ceil(400*(((self.s-3)**2.5)*math.log(self.s)))
		self.n_zeros = n_zeros
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
		self.corr = np.abs(np.dot(np.transpose(self.Y),self.Y))
		if verbose:
			print(f"M = {self.M}, K = {self.K}, N = {self.N}, s = {self.s}, default thresh = {round(self.default_thresh,2)}")

	def build_D(self):
		D = np.random.normal(0,1,(self.M,self.K))
		D = D/np.linalg.norm(D, axis = 0)
		return(D)

	def build_X(self):
		X = np.zeros((self.K,self.N))
		X[0,list(range(self.n_zeros))] = 1
		X[list(range(self.s)),0] = 1
		for i in range(1, self.n_zeros):
			start = 1+(self.s-1)*i
			rows = list(range(start,start + self.s - 1))
			X[rows,i] = 1
		for i in range(self.n_zeros,self.N):
			rows = random.sample(range(self.K),self.s)
			X[rows,i] = 1
		X = X - 2*X*np.random.binomial(1,0.5,(self.K,self.N))
		return(X)
	
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

