# spectral_dict_learn
Dictionary learning using spectral methods
##########################

This is a brief summary of the functionality of the subspace dictionary learning github package. This package includes a complete implementation of the methods described in [todo: link arxiv], along with tools for running and evaluating performance on simulated data.

This readme includes step-by-step instructions for reading and interpreting a standard simulation. The main test script is all_test_script.py, which generates a dictionary and sample data, recovers subspaces up to a specified number, and performs subspace intersection on the recovered subspaces.

**To ensure dictionary elements appear in subspace intersection, the sparsity pattern generated in all_test_script.py will automatically fix certain dictionary elements. Each consecutive pair of numbers will share the next dictionary element, starting at 0. This means if the test is run on 10 subspaces, y0 and y1 will contain dict element 0, y2 and y3 will contain dict element 1, and so on.**

#### RUNNING THE SCRIPT ####
To run all_test_script.py, we use the following sample command. Detailed parameter descriptions can be found in all_test_script.py, or by using the command: "python all_test_script.py -h".

python all_test_script.py -M 500 -K 1000 -s 8 -N 30000 -n_subspaces 10 -max_idx -1 -seed 123 -T 5 -tau 0.5

This runs the test suite with data dimension 500, 1000 dictionary elements, sparsity 8, and 30000 samples. The setting -n_subspaces 10 means that only the first 10 spanning subspaces will be estimated, and -T 5 means this test will be repeated and averaged over 5 different random dictionaries and samples. The seed is set to 123 for reproducibility.

Results are stored by default in a "results" directory in the current active directory. The directory will be created by the script if it does not already exist. This can be changed using the -result_dir parameter. Detailed results are stored in the files, while topline numbers can be found in the log file (e.g. average error, inner products).

By setting -oracle_mode, after the first pass is complete, the algorithm will rerun using the recovered dictionary elements as oracles. This is compared to the performance if the true dictionary element itself is used as oracle.

#### INTERPRETING RESULTS ####
When the simulation is completed, the log file will conclude with the following statistics:

2022-06-01 17:15:37,326 - all_test_script.py - INFO - Testing completed.
Avg recovery error: 0.230157
Avg inner product: 0.973366
Proportion of false recoveries: 0.000000

These should be understood as follows:
Avg recovery error: on the first (non-oracle) pass through the data, the average Euclidean distance between recovered vectors and the true corresponding dictionary vector.
Avg inner product: on the first (non-oracle) pass through the data, the average inner product between recovered vectors and the true corresponding dictionary vector.
Proportion of false recoveries: the proportion of false positive/false negative dictionary elements in the recovered dictionary elements. A false positive occurs when an estimated dictionary element is recovered even though the true subspaces do not have a one-dimensional intersection. A false negative occurs when no estimated dictionary element is returned, even though the true subspaces share one-dimensional intersection.

If oracle_mode is set, you will also see:
2022-06-01 17:15:37,326 - all_test_script.py - INFO - Testing completed.
Avg recovery error: 0.230157
Avg inner product: 0.973366
Proportion of false recoveries: 0.000000
Est Oracle avg inner: 0.988719
True Oracle avg inner: 0.984918

These statistics are:
Est Oracle avg inner: average inner product when the vector recovered in the first pass is used as oracle.
True Oracle avg inner: average inner product when the true dictionary element is used as oracle.

