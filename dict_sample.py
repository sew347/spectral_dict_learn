import numpy as np
import random
import math
import pickle
import argparse

class dict_sample:
	def __init__(self, M = None, K = None, N = None, s = None, verbose = None, n_zeros = 2):
		if M is None and s is None:
			raise AttributeError("Either M or s must not be None.")
		
		self.M = M if M is not None else math.ceil((50*s)**(1/0.95))
		#self.s = s if s is not None else math.ceil(self.M/10/math.log(self.M))
		self.s = s if s is not None else math.ceil(self.M/50)
		self.K = K if K is not None else math.ceil(5*(self.M**1))
		self.N = N if N is not None else math.ceil(5*(self.K**1.1))
		self.n_zeros = n_zeros
		self.default_thresh = 1/(self.s**2)
		if verbose:
			print(f"M = {self.M}, K = {self.K}, N = {self.N}, s = {self.s}, thresh = {round(self.default_thresh,2)}")
		self.D = self.build_D()
		self.X = self.build_X()
		self.Y = np.dot(self.D,self.X)
		self.Y = self.Y/np.linalg.norm(self.Y, axis = 0)

	def build_D(self):
		D = np.random.normal(0,1,(self.M,self.K))
		D = D/np.linalg.norm(D, axis = 0)
		return(D)

	def build_X(self):
		X = np.zeros((self.K,self.N))
		X[0,list(range(self.n_zeros))] = 1
		X[list(range(self.s)),0] = 1
		for i in range(1, self.n_zeros):
			rows = random.sample(range(1,self.K),self.s-1)
			X[rows,i] = 1
		for i in range(self.n_zeros,self.N):
			rows = random.sample(range(self.K),self.s)
			X[rows,i] = 1
		X = X - 2*X*np.random.binomial(1,0.5,(self.K,self.N))
		return(X)

