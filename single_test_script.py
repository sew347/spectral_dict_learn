import numpy as np
import random
import numpy.linalg as la
import argparse
import dict_sample as ds
import matplotlib.pyplot as plt
import time
from datetime import datetime
import logging
import pathlib
import csv
import single_subspace_recovery as ssr
import single_subspace_intersection as ssi

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="This script runs a randomized dictionary learning test.", \
	 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#	parser.add_argument('outfile', help='Output filepath')
	parser.add_argument('-M', help='Dimension of sample vectors', type = int)
	parser.add_argument('-N', help='Number of samples', default = None, type = int)
	parser.add_argument('-K', help='Number of dictionary elements', default = None, type = int)
	parser.add_argument('-s', help='Sparsity', default = None, type = int)
	parser.add_argument('-thresh', help='Thresholding parameter', default = 1/2, type = float)
	parser.add_argument('-T', help='Number of runs', default = 1, type = int)
	parser.add_argument('-seed', help='Random seed for test', default = None, type = int)
	args = parser.parse_args()
	
	#Filename for results
	save_results = True
	now = datetime.now()
	date_time = now.strftime("%m_%d_%Y_%H_%M")
	result_path = 'results/test_results_%(ts)s' % {'ts': date_time}
	pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)
	if save_results:
		res_fp = result_path + '/results.csv'
		arg_fp = result_path + '/args.csv'
		#save arguments:
		with open(arg_fp, 'w') as arg_f:
			writer = csv.writer(arg_f)
			arg_headers = ['M','s','K','N','thresh','T','seed']
			row = [args.M,args.s,args.K,args.N,args.thresh,args.T,args.seed]
			writer.writerow(arg_headers)
			writer.writerow(row)
		with open(res_fp, 'w') as res_f:
			writer = csv.writer(res_f)
			res_headers = ['Norm diff','Basis 0 diff','Basis 1 diff','sim time','est time']
			writer.writerow(res_headers)

	if args.seed is not None:
		np.random.seed(args.seed)
		random.seed(args.seed)
	
	logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
	
	
	for t in range(args.T):
		start = time.time()
		DS = ds.dict_sample(args.M,args.s,args.K,args.N, n_zeros = 2)
		sim_end = time.time()
		SSR0 = ssr.single_subspace_recovery(DS,args.thresh,0)
		SSR1 = ssr.single_subspace_recovery(DS,args.thresh,1)
		SSI = ssi.single_subspace_intersection(SSR0.S,SSR1.S)
		oracle = SSI.oracle
		est_end = time.time()
		D0 = DS.D[:,0]
		accu = np.min([la.norm(oracle - D0), la.norm(oracle+D0)])
		sim_time = sim_end - start
		est_time = est_end - sim_end
		if save_results:
			with open(res_fp, 'a') as res_f:
				writer = csv.writer(res_f)
				row = [accu, SSR0.eval_basis(), SSR1.eval_basis(), sim_time, est_time]
				writer.writerow(row)