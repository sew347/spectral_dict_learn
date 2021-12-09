import numpy as np
import random
import math
import pickle
import dict_sample as ds
import subspace_proj as sp

class oracle_finder:
	def __init__(self, DS, def_thresh = None):
		self.DS = DS
		self.def_thresh = def_thresh if def_thresh is not None else 1/(self.DS.s**2)
		self.C = self.build_C()
		self.C_norm = self.C/np.linalg.norm(self.C)
		self.As = {}
		self.bases = {}

	#approx covariance matrix
	def build_C(self):
		C = np.zeros((self.DS.M,self.DS.M))
		for i in range(self.DS.N):
			C = C + np.outer(self.DS.Y[:,i],self.DS.Y[:,i])
		return C/self.DS.N

	def build_Ai(self,i,thresh=None):
		if thresh == None:
			thresh = self.def_thresh
		A = np.outer(self.DS.Y[:,i],self.DS.Y[:,i])
		Afp = np.zeros((self.DS.M,self.DS.M))
		corr_idx =[]
		all_corrs = []
		match_corrs = []
		supp = np.nonzero(self.DS.X[:,i])
		fn = 0
		fp = 0
		if i == 0:
			idx = list(range(1,self.DS.N))
		else:
			idx = list(range(i-1))+list(range(i+1,self.DS.N))
		for j in idx:
			cr = np.inner(self.DS.Y[:,i],self.DS.Y[:,j])**2
			all_corrs.append(cr)
			supp_j = np.nonzero(self.DS.X[:,j])
			supp_j_nomatch = np.setdiff1d(supp_j,supp)
			if len(np.intersect1d(supp_j,supp)) > 0:
				match_corrs.append(cr)
			if cr > thresh:
				corr_idx.append(j)
				A = A + np.outer(self.DS.Y[:,j],self.DS.Y[:,j])
				# resid = np.zeros((self.DS.M,))
				# for k in supp_j_nomatch:
				# 	#if j == 1: print(j,k,self.DS.X[k,j])
				# 	resid = resid + self.DS.X[k,j]*self.DS.D[:,k]
				# resid = resid/(np.linalg.norm(np.dot(self.DS.D,self.DS.X[:,j])))
				# Afp = Afp + np.outer(resid, resid)
		A = A/(len(corr_idx)+1)
		Afp = Afp/(len(corr_idx)+1)
		#print(np.mean(all_corrs), np.mean(match_corrs))
		return A, corr_idx, Afp

	def build_Ai_adj(self,i):
		A, corr_idx, Afp = self.build_Ai(i)
		inner_prod = np.trace(np.dot(A,self.C_norm))
		return A - inner_prod*self.C_norm

	def get_basis(self,A):
		E = np.linalg.eigh(A)
		return E[1][:,self.DS.M-self.DS.s:]

	def svd_mat(self,A):
		dims = np.shape(A)
		A_out = np.outer(A[:,0],A[:,0])
		for i in range(1,dims[1]):
			A_out = A_out + np.outer(A[:,i],A[:,i])
		return(A_out)

	def merge_basis(self,A,B):
		A_out = self.svd_mat(A)
		B_out = self.svd_mat(B)
		#E = np.linalg.eigh(A_out+B_out)
		#print(E[0][self.DS.M-self.DS.s:])
		return(np.linalg.eigh(A_out+B_out)[1][:,self.DS.M-1])

	def true_oracle(self, i):
		idx = np.nonzero(self.DS.X[i,:])
		n_idx = np.shape(idx)[1]
		idx = np.reshape(idx,(n_idx,))
		Oi = np.sum(np.dot(self.DS.D,self.DS.X[i,idx]*self.DS.X[:,idx]), axis = 1)
		Oi = Oi/np.linalg.norm(Oi)
		return Oi




