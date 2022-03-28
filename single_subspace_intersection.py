import numpy as np
import numpy.linalg as la

class single_subspace_intersection:

	def __init__(self, SR, i,j, delta = 0.5):
		self.i = i
		self.j = j
		self.Si = SR.subspaces[i].S
		self.Sj = SR.subspaces[j].S
		self.delta = delta
		self.M = np.shape(self.Si)[0]
		self.dhat, self.emp_uniq_int_flag = self.subspace_intersection()
		self.d, self.true_uniq_int_flag, self.true_uniq_int_idx = self.true_subspace_intersection(SR.DS)
		self.accu = self.eval_intersection()

	def proj_mat(self, A):
		return np.dot(A,np.transpose(A))

	def subspace_intersection(self):
		Pj = self.proj_mat(self.Sj)
		P_itoj = self.Si - np.dot(Pj,self.Si)
		SVD = la.svd(P_itoj)
		sing_vals = SVD[1]
		if (sing_vals[-1] < self.delta) and (sing_vals[-2] >= self.delta):
			return (np.dot(self.Si, SVD[2][-1]),True)
		else:
			return (0,False)
		
	def true_subspace_intersection(self,DS):
		supp_i = np.nonzero(DS.X[:,self.i])
		supp_j = np.nonzero(DS.X[:,self.j])
		supp_int = np.intersect1d(supp_i,supp_j)
		if np.shape(supp_int)[0]==1 == 1:
			return (DS.D[:,supp_int[0]], True, supp_int[0])
		else:
			return (0, False, -1/2)
		
	def eval_intersection(self):
		if not(self.emp_uniq_int_flag) and not(self.true_uniq_int_flag):
			return 0
		elif not(self.emp_uniq_int_flag) and self.true_uniq_int_flag:
			return -1
		elif self.emp_uniq_int_flag and not(self.true_uniq_int_flag):
			return 1
		else:
			return np.min([la.norm(self.d - self.dhat),la.norm(self.d + self.dhat)])