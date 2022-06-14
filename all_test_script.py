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
import oracle_recovery as orr

# main testing script for dictionary learning

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="This script runs a randomized dictionary learning test.", \
	 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-s', help='Sparsity', default = None, type = int)
	parser.add_argument('-M', help='Dimension of sample vectors', type = int)
	parser.add_argument('-N', help='Number of samples', default = None, type = int)
	parser.add_argument('-K', help='Number of dictionary elements', default = None, type = int)
	parser.add_argument('-thresh', help='Thresholding parameter for mode = thresh', default = 1/2, type = float)
	parser.add_argument('-tau', help='Singular value threshold', default = 1/2, type = float)
	parser.add_argument('-distribution', help='distribution', default = 'bernoulli', type = str)
	parser.add_argument('-mode', help='Subspace recovery method; supported options are quad_weight for E<yj,yi>^2 yiyi^T, thresh for thresholded yiyi^T.', default = 'quad_weight', type = str)
	parser.add_argument('-T', help='Number of dictionaries to test over', default = 1, type = int)
	parser.add_argument('-n_subspaces', help='Number of subspaces to recover per dictionary', default = 10, type = int)
	parser.add_argument('-max_idx', help = 'Maximum index for subspace intersection', default = 10, type = int)
	parser.add_argument('-no_oracle', help='Flag for not including oracle recovery comparison', action='store_false')
	parser.add_argument('-lowmem', help='Set to reduce memory overhead',action='store_true')
	parser.add_argument('-n_processes', help = 'DEPRECATED: kept to rerun old shell commands', type = int)
	parser.add_argument('-result_dir', help='Destination directory for output files', default = 'results', type = str)
	parser.add_argument('-logflag', help='Flag for additional logging', action='store_true')
	parser.add_argument('-seed', help='Random seed for test', default = None, type = int)
	args = parser.parse_args()
	
	M = args.M
	N = args.N
	K = args.K
	s = args.s
	mode = args.mode
	thresh = args.thresh
	tau = args.tau
	distribution = args.distribution
	T = args.T
	n_subspaces = N if args.n_subspaces == -1 else args.n_subspaces
	max_idx = args.max_idx
	oracle_flag = args.no_oracle
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
		res_fp = result_path + '/results.csv'
		arg_fp = result_path + '/args.csv'
		#save arguments:
		with open(arg_fp, 'w') as arg_f:
			writer = csv.writer(arg_f)
			arg_headers = ['M','s','K','N','mode','thresh','tau','distribution',\
						   'T','n_subspaces','max_idx','lowmem','result_dir','seed']
			row = [M,s,K,N,mode,thresh,tau,distribution,T,n_subspaces,max_idx,\
				   lowmem,result_dir,seed]
			writer.writerow(arg_headers)
			writer.writerow(row)
		with open(res_fp, 'w') as res_f:
			writer = csv.writer(res_f)
			res_headers = ['t','i','j','error','inner','Si error','Sj error','emp_uniq_int_flag','true_int_flag','true_int_idx','sim time','est time']
			writer.writerow(res_headers)
		logging.info('Arguments and results saving to ' + result_path)
	
	#ensure test range will include significant number of overlaps
	fixed_supp = []
	for i in range(int(n_subspaces/2)):
		fixed_supp += [i,i]

	logging.info('Beginning testing.')
	all_errs = []
	all_inners = []
	total_false_count = 0
	total_n_ints = 0
	all_orr_true_inners = []
	all_orr_est_inners = []
	for t in range(args.T):
		start = time.time()
		DS = ds.dict_sample(M,s,K,N, distribution = distribution, lowmem=lowmem, thresh=thresh, n_subspaces = n_subspaces, fixed_supp = fixed_supp)
		sim_end = time.time()
		logging.info('Dictionary %d generated in time %d' % (t+1,sim_end - start))
		SR = sr.subspace_recovery(DS, n_subspaces, mode = mode)
		sub_end = time.time()
		logging.info('Subspace recovery batch %d recovered in time %d with avg subspace error %2f' % (t+1, sub_end-sim_end, np.mean(SR.errs)))
		SI = si.subspace_intersection(SR, tau = tau, max_idx = max_idx)
		est_end = time.time()
		logging.info('Subspace intersection batch %d recovered in time %d' % (t+1, est_end - sub_end))
		sim_time = sim_end - start
		est_time = est_end - sim_end
		logging.info('Total test time for batch %d: %d sec' % (t+1, est_time))
		if save_results:
			with open(res_fp, 'a') as res_f:
				writer = csv.writer(res_f)
				for SI_i in SI.intersections:
					for SSI in SI_i:
						row =[t,SSI.i,SSI.j,SSI.err,SSI.inner,SSI.err_Si,SSI.err_Sj,\
						  SSI.emp_uniq_int_flag, SSI.true_uniq_int_flag,SSI.true_uniq_int_idx,sim_time,est_time]
						writer.writerow(row)
		all_errs = all_errs + SI.errs
		all_inners = all_inners + SI.inners
		total_false_count = total_false_count + SI.false_count
		total_n_ints = total_n_ints + SI.n_ints
		logging.info('Batch %d completed with avg inner product %2f'%(t+1,np.mean(SI.inners)))
		if oracle_flag:
			ORR_est = orr.oracle_recovery(DS,SI = SI)
			all_orr_est_inners = all_orr_est_inners + ORR_est.inners
			logging.info('Est oracle %d completed with avg inner product %2f'%(t+1,np.mean(ORR_est.inners)))
			ORR_true = orr.oracle_recovery(DS, n_cols = int(n_subspaces/2))
			all_orr_true_inners = all_orr_true_inners + ORR_true.inners
			logging.info('True oracle %d completed with avg inner product %2f'%(t+1,np.mean(ORR_true.inners)))
	if oracle_flag:
		logging.info('Testing completed.\nAvg recovery error: %2f\nAvg inner product: %2f\nProportion of false recoveries: %2f\nEst Oracle avg inner: %2f\nTrue Oracle avg inner: %2f'%(np.mean(all_errs),np.mean(all_inners),(total_false_count/total_n_ints),np.mean(all_orr_est_inners),np.mean(all_orr_true_inners)))
	else:
		logging.info('Testing completed.\nAvg recovery error: %2f\nAvg inner product: %2f\nProportion of false recoveries: %2f'%(np.mean(all_errs),np.mean(all_inners),(total_false_count/total_n_ints)))