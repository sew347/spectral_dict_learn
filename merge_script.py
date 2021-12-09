import numpy as np
import random
import numpy.linalg as la
import argparse
import dict_sample as ds
import oracle_finder as of
import subspace_proj as sp
import matplotlib.pyplot as plt
import time

def main():
	parser = argparse.ArgumentParser(description="This script runs a randomized dictionary learning test.", \
	 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-outfile', help='Output filepath')
	parser.add_argument('-M', help='Dimension of sample vectors', default = None, type = int)
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

	accus = []
	controls = []
	oracle_ratios = []
	true_accus = []
	for t in range(args.T):
		start = time.time()
		DS = ds.dict_sample(args.M,args.K,args.N,args.s,True) if t == 0 else ds.dict_sample(args.M,args.K,args.N,args.s)
		OF = of.oracle_finder(DS, def_thresh = args.thresh)
		# A0, corr_idx, Afp = OF.build_Ai(0)
		# A1, corr_idx, Afp = OF.build_Ai(1)
		A0 = OF.build_Ai_adj(0)
		A1 = OF.build_Ai_adj(1)

		control0 = abs(np.inner(DS.D[:,0], DS.Y[:,0]))
		control1 = abs(np.inner(DS.D[:,0], DS.Y[:,1]))
		controls.append(control0)
		controls.append(control1)


		bas0 = OF.get_basis(A0)
		bas1 = OF.get_basis(A1)

		est = OF.merge_basis(bas0,bas1)
		end = time.time()
		oracle_accu = abs(np.inner(est, DS.D[:,0]))
		accus.append(oracle_accu)
		oracle_ratios.append(oracle_accu/control0)
		oracle_ratios.append(oracle_accu/control1)

		true_accu = abs(np.inner(OF.true_oracle(0),DS.D[:,0]))
		true_accus.append(true_accu)

		teststring = "Test "+str(t+1)+": "+str(round(oracle_accu,4))+" vs " + str(round(true_accu,4)) + " in time "+str(round((end-start),2))+" sec"
		print(teststring)

	print(round(np.mean(accus)*DS.M/DS.s/17.5,4),round(np.mean(accus),4), round(np.mean(true_accus),4), round(np.mean(accus)/np.mean(true_accus),4), round(np.mean(controls),4), round(np.mean(oracle_ratios),4), round(np.min(oracle_ratios),4))

if __name__ == "__main__":
	main()