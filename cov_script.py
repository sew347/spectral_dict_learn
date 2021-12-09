import numpy as np
import random
import numpy.linalg as la
import argparse
import dict_sample as ds
import oracle_finder as of
import subspace_proj as sp
import matplotlib.pyplot as plt

def main():
	parser = argparse.ArgumentParser(description="This script runs a randomized dictionary learning test.", \
	 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('outfile', help='Output filepath')
	parser.add_argument('M', help='Dimension of sample vectors', type = int)
	parser.add_argument('-N', help='Number of samples', default = None, type = int)
	parser.add_argument('-K', help='Number of dictionary elements', default = None, type = int)
	parser.add_argument('-s', help='Sparsity', default = None, type = int)
	parser.add_argument('-thresh', help='Thresholding parameter', default = None, type = float)
	parser.add_argument('-e','--epsi', help='Noise level', default = 0, type = float)
	parser.add_argument('-T', help='Number of runs', default = 1, type = int)
	parser.add_argument('-seed', help='Random seed for test', default = None, type = int)
	parser.add_argument('-v','--verbose', help='Flag to print output', default = None, action = 'store_true')
	args = parser.parse_args()

	if args.seed is not None:
		np.random.seed(args.seed)
		random.seed(args.seed)

	results = []
	for t in range(args.T):
		DS = ds.dict_sample(args.M,args.K,args.N,args.s,True) if t == 0 else ds.dict_sample(args.M,args.K,args.N,args.s)
		OF = of.oracle_finder(DS)
		A, corr_idx, Afp = OF.build_Ai(0)
		# E = la.eigh(A)
		# plt.hist(E[0], density = True, bins=40)
		# plt.ylabel('Count')
		# plt.xlabel('Data')
		# plt.show()

		basis = OF.get_basis(A)
		proj_norms = []
		for i in range(DS.s):
			proj, resid = sp.subspace_proj(DS.D[:,i],basis)
			proj_norms.append(np.linalg.norm(proj))
		result = np.mean(proj_norms)
		results.append(result)
		teststring = "Test "+str(t+1)+": "+str(round(result,4))#+" in time "+str(round((end-start)/60,2))+" min"
		if args.verbose:
			print(teststring)
	print(np.mean(results))

if __name__ == "__main__":
	main()
