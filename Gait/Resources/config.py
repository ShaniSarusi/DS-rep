from os.path import join, sep
from os import getcwd

n_folds = 5
max_evals = 4
opt_alg = 'random'  # Can be 'tpe' or 'random'
do_verbose = False
tasks_to_optimize = 'both_split_and_all'

algs = ['lhs', 'rhs', 'overlap', 'overlap_strong', 'combined']
#algs = ['overlap_strong']

# metric_to_optimize = 'sc_rmse'
# search_space = 'param9'
# outlier_percent_to_remove = 5

metric_to_optimize = 'asym_rmse'
search_space = 'param_asym_1'
outlier_percent_to_remove = 1


#################################################################################
exp = 2  # can be either exp 1 or 2 for now
run_on_cloud = False
if 'hadoop' in getcwd():
    run_on_cloud2 = True
    do_multi_core = True
else:
    run_on_cloud2 = False
    do_multi_core = False
###################################################################################
# Paths
local_windows_path = join('C:', sep, 'Users', 'zwaks', 'Documents', 'Data')
s3_path = 'data'
aws_region_name = 'us-west-2'
s3_bucket = 'intel-health-analytics'

if run_on_cloud:
    data_path = s3_path
elif run_on_cloud2:
    data_path = '/home/hadoop/Zeev'
else:
    data_path = local_windows_path
if exp == 1:
    common_path = join(data_path, 'Intel APDM March 6 2017')
elif exp == 2:
    common_path = join(data_path, 'APDM June 2017')
pickle_path = join(common_path, 'Pickled')
input_path = join(common_path, 'Subjects')
results_path = join(common_path, 'Results')

####################################################################
# Parameters and constants
G = 9.80665  # standard gravity
radian = 57.2958
sampling_rate = 128
if exp == 1:
    lhs_wrist_sensor = 377  # 1st exp values
    rhs_wrist_sensor = 638  # 1st exp values
elif exp == 2:
    lhs_wrist_sensor = 1589  # 2nd exp values
    rhs_wrist_sensor = 1695  # 2nd exp values
    lhs_leg_sensor = 377
    rhs_leg_sensor = 638
    lumbar_sensor = 1429
    trunk_sensor = 793
