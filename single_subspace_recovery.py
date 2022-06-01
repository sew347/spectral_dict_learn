import numpy as np
import random
import math
import pickle
import dict_sample as ds
import logging
import time

class single_subspace_recovery:
	def __init__(self, DS, i, mode = 'corr_weight', thresh = 1/2):
		self.mode = mode
		self.i = i
		self.M = DS.M
		self.s = DS.s
		if DS.lowmem:
			self.I_i = np.setdiff1d(np.arange(DS.N),DS.uncorr_idx[i])
		else:
			self.I_i = np.nonzero(DS.corr[i,:] > thresh)[0]
		self.Ni = np.shape(self.I_i)[0]
		HSig_i = self.build_HSig_i(DS)
		HSig_proj_i = self.build_HSig_proj_i(DS,HSig_i)
		self.S = self.get_basis(HSig_proj_i)
		self.err = self.eval_basis(DS)

	def build_HSig_i(self, DS):
		if self.mode == 'thresh':
			Y_I_i = DS.Y[:,self.I_i]
			HSig_i = np.dot(Y_I_i,np.transpose(Y_I_i))
			HSig_i = HSig_i/len(self.I_i)
		if self.mode == 'corr_weight':
			weights = DS.corr[self.i,:]
			mu = np.mean(weights)
			#WY = (weights/mu)*DS.Y[:,idx]
			WY = (weights/mu)*DS.Y
			HSig_i = np.dot(WY,np.transpose(WY))/DS.N
		else:
			#This should really be an enum
			raise ValueError('Invalid mode for subspace recovery. Currently supported modes are thresh and corr_weight.')
		return HSig_i
	
	def increment_outer(self, HSig, DS, j):
		return HSig + np.outer(DS.Y[:,j],DS.Y[:,j])

	def build_HSig_proj_i(self,DS, HSig_i):
		inner_prod = np.trace(np.dot(HSig_i,DS.HSig_D))
		return HSig_i - inner_prod*DS.HSig_D

	def get_basis(self, HSig_proj_i):
		E = np.linalg.eigh(HSig_proj_i)
		return E[1][:,self.M-self.s:]
	
	def subspace_dist(self,A,B):
		P_B = np.dot(B,np.transpose(B))
		P_AtoB = A - np.dot(P_B,A)
		return(np.linalg.norm(P_AtoB,2))
	
	def eval_basis(self, DS):
		supp = np.nonzero(DS.X[:,self.i])[0]
		true_basis = np.linalg.qr(DS.D[:,supp])[0]
		return self.subspace_dist(true_basis,self.S)
