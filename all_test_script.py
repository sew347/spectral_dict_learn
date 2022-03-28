import numpy as np
import random
import numpy.linalg as la
import argparse
import dict_sample as ds
import matplotlib.pyplot as plt
import time
from datetime import datetime
import logging
import os
import pathlib
import csv
import subspace_recovery as sr
import subspace_intersection as si

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="This script runs a randomized dictionary learning test.", \
	 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#	parser.add_argument('outfile', help='Output filepath')
	parser.add_argument('-s', help='Sparsity', default = None, type = int)
	parser.add_argument('-M', help='Dimension of sample vectors', type = int)
	parser.add_argument('-N', help='Number of samples', default = None, type = int)
	parser.add_argument('-K', help='Number of dictionary elements', default = None, type = int)
	parser.add_argument('-thresh', help='Thresholding parameter', default = 1/2, type = float)
	parser.add_argument('-delta', help='Singular value threshold', default = 1/2, type = float)
	parser.add_argument('-T', help='Number of runs', default = 1, type = int)
	parser.add_argument('-n_subspaces', help='Number of subspaces per dictionary', default = 10, type = int)
	parser.add_argument('-max_idx', help = 'Maximum index for subspace intersection', default = 10, type = int)
	parser.add_argument('-n_processes', help='Number of parallel processes', default = 1, type = int)
	parser.add_argument('-result_dir', help='Destination directory for output files', default = 'results', type = str)
	parser.add_argument('-seed', help='Random seed for test', default = None, type = int)
	args = parser.parse_args()
	
	M = args.M
	N = args.N
	K = args.K
	s = args.s
	thresh = args.thresh
	delta = args.delta
	T = args.T
	n_subspaces = N if args.n_subspaces == -1 else args.n_subspaces
	max_idx = args.max_idx
	n_processes = args.n_processes
	parallel_flag = True if n_processes >= 2 else False
	result_dir = args.result_dir
	seed = args.seed
	
	if args.seed is not None:
		np.random.seed(args.seed)
		random.seed(args.seed)
	#Filename for results
	save_results = True
	now = datetime.now()
	date_time = now.strftime("%m_%d_%Y_%H_%M")
	result_path = result_dir + '/test_results_%(ts)s' % {'ts': date_time}
	result_path = os.path.abspath(result_path)
	pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)
	log_fp = result_path + '/logs.log'
	logging.basicConfig(filename=log_fp, level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
	if save_results:
		logging.info
		res_fp = result_path + '/results.csv'
		arg_fp = result_path + '/args.csv'
		#save arguments:
		with open(arg_fp, 'w') as arg_f:
			writer = csv.writer(arg_f)
			arg_headers = ['M','s','K','N','thresh','delta','T','n_subspaces',\
						   'max_idx','n_processes','result_dir','seed']
			row = [M,s,K,N,thresh,delta,T,n_subspaces,max_idx,n_processes,result_dir,seed]
			writer.writerow(arg_headers)
			writer.writerow(row)
		with open(res_fp, 'w') as res_f:
			writer = csv.writer(res_f)
			res_headers = ['t','i','j','accu','emp_int_flag','true_int_flag','true_int_idx','sim time','est time']
			writer.writerow(res_headers)
		logging.info('Arguments and results saving to ' + result_path)
	
	logging.info('Beginning testing.')
	for t in range(args.T):
		start = time.time()
		DS = ds.dict_sample(M,s,K,N, n_zeros = 1)
		sim_end = time.time()
		SR = sr.subspace_recovery(DS, thresh, n_subspaces, parallel = parallel_flag, n_processes = n_processes)
		SI = si.subspace_intersection(SR, delta = delta, max_idx = max_idx, parallel = parallel_flag, n_processes = n_processes)
		est_end = time.time()
		sim_time = sim_end - start
		est_time = est_end - sim_end
		if save_results:
			with open(res_fp, 'a') as res_f:
				writer = csv.writer(res_f)
				for SSI in SI.intersections:
					row =[t,SSI.i,SSI.j,SSI.accu,SSI.emp_uniq_int_flag,\
						  SSI.true_uniq_int_flag,SSI.true_uniq_int_idx,sim_time,est_time]
					writer.writerow(row)
	logging.info('Testing completed.')