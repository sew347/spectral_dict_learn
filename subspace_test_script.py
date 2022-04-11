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
import cProfile
import pstats

def main():
	parser = argparse.ArgumentParser(description="This script runs a randomized dictionary learning test.", \
	 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-s', help='Sparsity', default = None, type = int)
	parser.add_argument('-M', help='Dimension of sample vectors', type = int)
	parser.add_argument('-N', help='Number of samples', default = None, type = int)
	parser.add_argument('-K', help='Number of dictionary elements', default = None, type = int)
	parser.add_argument('-thresh', help='Thresholding parameter', default = 1/2, type = float)
	parser.add_argument('-T', help='Number of runs', default = 1, type = int)
	parser.add_argument('-n_subspaces', help='Number of subspaces per dictionary', default = 10, type = int)
	parser.add_argument('-n_processes', help='Number of parallel processes', default = 1, type = int)
	parser.add_argument('-lowmem', help='Set to reduce memory overhead',action='store_true')
	parser.add_argument('-result_dir', help='Destination directory for output files', default = 'results', type = str)
	parser.add_argument('-seed', help='Random seed for test', default = None, type = int)
	parser.add_argument('-profile', help='Flag to profile runtime',action='store_true')
	args = parser.parse_args()
	
	M = args.M
	N = args.N
	K = args.K
	s = args.s
	thresh = args.thresh
	T = args.T
	n_subspaces = N if args.n_subspaces == -1 else args.n_subspaces
	n_processes = args.n_processes
	lowmem = args.lowmem
	result_dir = args.result_dir
	seed = args.seed
	profile = args.profile
	
	if profile:
		profiler = cProfile.Profile()
		profiler.enable()
	
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
		logging.info('Saving arguments.')
		res_fp = result_path + '/results.csv'
		arg_fp = result_path + '/args.csv'
		avg_fp = result_path + '/avg.csv'
		#save arguments:
		with open(arg_fp, 'w') as arg_f:
			writer = csv.writer(arg_f)
			arg_headers = ['M','s','K','N','thresh','T','n_subspaces','n_processes','lowmem','seed']
			row = [args.M,args.s,args.K,args.N,args.thresh,args.T,args.n_subspaces, args.n_processes, args.lowmem,args.seed]
			writer.writerow(arg_headers)
			writer.writerow(row)
		with open(res_fp, 'w') as res_f:
			writer = csv.writer(res_f)
			res_headers = ['t','i','basis error','sim time','est time']
			writer.writerow(res_headers)
		logging.info('Arguments and results saving to ' + result_path)
	
	logging.info('Beginning testing.')
	all_errs = []
	for t in range(args.T):
		start = time.time()
		DS = ds.dict_sample(M,s,K,N,n_processes = n_processes, lowmem=lowmem, thresh=thresh, n_subspaces = n_subspaces)
		sim_end = time.time()
		logging.info('Dictionary %d generated in time %d' % (t+1,sim_end - start))
		SR = sr.subspace_recovery(DS, thresh, n_subspaces, n_processes = n_processes)
		est_end = time.time()
		logging.info('Subspace batch %d recovered in time %d with avg error %2f' % (t+1, est_end - sim_end, np.mean(SR.errs)))
		sim_time = sim_end - start
		est_time = est_end - sim_end
		all_errs = all_errs + SR.errs
		if save_results:
			with open(res_fp, 'a') as res_f:
				writer = csv.writer(res_f)
				for i in range(n_subspaces):
					row = [t, i, SR.subspaces[i].err, sim_time, est_time]
					writer.writerow(row)
	logging.info('Testing completed. Avg error %2f'%np.mean(all_errs))

	if profile:
		profiler.disable()
		stats = pstats.Stats(profiler).sort_stats('cumtime')
		stats.strip_dirs()
		stats.dump_stats(result_path+'/profile_data.prof')
		#rewrite to human-readable file
		with open(result_path+'/profile_data.txt', "w") as f:
			ps = pstats.Stats(result_path+'/profile_data.prof', stream=f)
			ps.sort_stats('cumtime')
			ps.print_stats()

if __name__ == "__main__":
	main()