import numpy as np
import random
import math
import pickle
import dict_sample as ds
import logging
import time

#single_subspace_recovery.py
#### summary ####
#runs subspace recovery algorithm using an oracle and compares to true dict element
#### inputs ####
#dictionary_sample DS, vector oracle, vector true_col
#### fields ####
#M,s = index, dimension, sparsity
#HSig_oracle = Correlation-weighted covariance matrix with oracle
#HSig_proj_oracle = Complement of Correlation-weighted covariance matrix with HSig
#dhat = resulting dictionary estimate
#inner = |<true_col,dhat>|

class single_oracle_recovery:
	def __init__(self, DS, oracle, true_col = None):
		self.M = DS.M
		self.s = DS.s
		self.oracle = oracle
		if true_col is None:
			self.true_col = self.oracle
		else:
			self.true_col = true_col
		HSig_oracle = self.build_HSig_oracle(DS)
		HSig_proj_oracle = self.build_HSig_proj_oracle(DS,HSig_oracle)
		self.dhat = self.get_col(HSig_proj_oracle)
		self.inner = np.abs(np.dot(self.dhat,self.oracle))

	def build_HSig_oracle(self, DS):
		weights = np.dot(np.transpose(self.oracle),DS.Y)
		mu = np.mean(weights)
		WY = (weights/mu)*DS.Y
		HSig_oracle = np.dot(WY,np.transpose(WY))/DS.N
		return HSig_oracle

	def build_HSig_proj_oracle(self,DS, HSig_oracle):
		inner_prod = np.trace(np.dot(HSig_oracle,DS.HSig_D))
		return HSig_oracle - inner_prod*DS.HSig_D

	def get_col(self, HSig_proj_oracle):
		E = np.linalg.eigh(HSig_proj_oracle)
		return E[1][:,self.M-1]
	
# 	def subspace_dist(self,A,B):
# 		P_B = np.dot(B,np.transpose(B))
# 		P_AtoB = A - np.dot(P_B,A)
# 		return(np.linalg.norm(P_AtoB,2))
	
# 	def eval_basis(self, DS):
# 		supp = np.nonzero(DS.X[:,self.i])[0]
# 		true_basis = np.linalg.qr(DS.D[:,supp])[0]
# 		return self.subspace_dist(true_basis,self.S)
