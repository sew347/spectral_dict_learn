import numpy as np
import numpy.linalg as la

class single_subspace_intersection:

	def __init__(self, S1, S2, delta = 0.5):
		self.S1 = S1
		self.S2 = S2
		self.delta = delta
		self.M = np.shape(self.S1)[0] 
		self.oracle = self.subspace_intersection()

	def proj_mat(self, A):
		return np.dot(A,np.transpose(A))

	def subspace_intersection(self):
		P2 = self.proj_mat(self.S2)
		P_1to2 = self.S1 - np.dot(P2,self.S1)
		SVD = la.svd(P_1to2)
		sing_vals = SVD[1]
		if (sing_vals[-1] < self.delta) and (sing_vals[-2] >= self.delta):
			return np.dot(self.S1, SVD[2][-1])
		else:
			return 0