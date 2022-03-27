import numpy as np
import random
import math
import pickle
import dict_sample as ds

class single_subspace_recovery:
	def __init__(self, DS, thresh, i):
		self.DS = DS
		self.i = i
		self.corr_i = DS.corr[:,i]
		self.I_i = np.nonzero(self.corr_i > thresh)[0]
		self.HSig_i = self.build_HSig_i()
		self.HSig_proj_i = self.build_HSig_proj_i()
		self.S = self.get_basis()

	def build_HSig_i(self):
		HSig_i = np.zeros((self.DS.M,self.DS.M))
		for j in self.I_i:
			HSig_i = HSig_i + np.outer(self.DS.Y[:,j],self.DS.Y[:,j])
		HSig_i = HSig_i/len(self.I_i)
		return HSig_i

	def build_HSig_proj_i(self):
		inner_prod = np.trace(np.dot(self.HSig_i,self.DS.HSig_D))
		return self.HSig_i - inner_prod*self.DS.HSig_D

	def get_basis(self):
		E = np.linalg.eigh(self.HSig_proj_i)
		return E[1][:,self.DS.M-self.DS.s:]
	
	def subspace_dist(self,A,B):
		P_B = np.dot(B,np.transpose(B))
		P_AtoB = A - np.dot(P_B,A)
		return(np.linalg.norm(P_AtoB,2))
	
	def eval_basis(self):
		supp = np.nonzero(self.DS.X[:,self.i])[0]
		true_basis = np.linalg.qr(self.DS.D[:,supp])[0]
		return self.subspace_dist(true_basis,self.S)
