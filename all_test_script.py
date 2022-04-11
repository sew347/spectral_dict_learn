import numpy as np
import random
import numpy.linalg as la
import argparse
import dict_sample as ds
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
	parser.add_argument('-lowmem', help='Set to reduce memory overhead',action='store_true')
	parser.add_argument('-result_dir', help='Destination directory for output files', default = 'results', type = str)
	parser.add_argument('-logflag', help='Flag for additional logging', action='store_true')
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
	lowmem = args.lowmem
	result_dir = args.result_dir
	seed = args.seed
	logflag = args.logflag
	
	if args.seed is not None:
		np.random.seed(args.seed)
		random.seed(args.seed)
	#Filename for results
	save_results = True
	now = datetime.now()
	date_time = now.strftime("%m_%d_%Y_%H_%M")
	argstring = 's%d_M%d_K%d_N%d_' % (s,M,K,N)
	result_path = result_dir + '/'+argstring+'%(ts)s' % {'ts': date_time}
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
						   'max_idx','n_processes','lowmem','result_dir','seed']
			row = [M,s,K,N,thresh,delta,T,n_subspaces,max_idx,n_processes,lowmem,result_dir,seed]
			writer.writerow(arg_headers)
			writer.writerow(row)
		with open(res_fp, 'w') as res_f:
			writer = csv.writer(res_f)
			res_headers = ['t','i','j','accu','true_int_flag','true_int_idx','sim time','est time']
			writer.writerow(res_headers)
		logging.info('Arguments and results saving to ' + result_path)
	
	#ensure test range will include significant number of overlaps
	fixed_supp = []
	for i in range(int(n_subspaces/2)):
		fixed_supp += [i,i]

	logging.info('Beginning testing.')
	for t in range(args.T):
		start = time.time()
		DS = ds.dict_sample(M,s,K,N, n_processes = n_processes, lowmem=lowmem, thresh=thresh, n_subspaces = n_subspaces, fixed_supp = fixed_supp)
		sim_end = time.time()
		logging.info('Dictionary %d generated in time %d' % (t+1,sim_end - start))
		SR = sr.subspace_recovery(DS, thresh, n_subspaces, n_processes = n_processes)
		sub_end = time.time()
		logging.info('Subspace recovery batch %d recovered in time %d with avg subspace error %2f' % (t+1, sub_end-sim_end, np.mean(SR.errs)))
		SI = si.subspace_intersection(SR, delta = delta, max_idx = max_idx, n_processes = n_processes)
		est_end = time.time()
		logging.info('Subspace intersection batch %d recovered in time %d' % (t+1, est_end - sub_end))
		sim_time = sim_end - start
		est_time = est_end - sim_end
		logging.info('Total test time for batch %d: %d sec' % (t+1, est_time))
		all_errs = []
		all_inners = []
		false_count = 0
		if save_results:
			with open(res_fp, 'a') as res_f:
				writer = csv.writer(res_f)
				for SI_i in SI.intersections:
					for SSI in SI_i:
						row =[t,SSI.i,SSI.j,SSI.err,\
						  SSI.true_uniq_int_flag,SSI.true_uniq_int_idx,sim_time,est_time]
						writer.writerow(row)
		for SI_i in SI.intersections:
			for SSI in SI_i:
				if SSI.true_uniq_int_flag and SSI.emp_uniq_int_flag:
					all_errs.append(SSI.err)
					all_inners.append(SSI.inner)
				else:
					false_count += 1
	logging.info('Testing completed. Avg recovery error %2f. Avg inner product %2f. Number of false recoveries: %d'%(np.mean(all_errs),np.mean(all_inners),false_count))