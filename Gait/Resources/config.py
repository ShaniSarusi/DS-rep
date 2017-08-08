from os.path import join, sep
from os import getcwd

machine = 0  # 1, 2, 3, 4, 5, 6
n_folds = 5
max_evals = 200
# algs = ['lhs', 'rhs', 'overlap', 'overlap_strong', 'combined']
# algs = ['lhs', 'rhs', 'overlap', 'overlap_strong']
algs = ['lhs', 'overlap', 'overlap_strong']
opt_alg = 'tpe'  # Can be 'tpe' or 'random'
metric_to_optimize = 'rmse'  # 'rmse' or 'mape'
do_verbose = False

data_type = 'both'
search_space = 'fast4' # or 5 ..

if machine == 1:
    data_type = 'both'
    search_space = 'fast2'
elif machine == 2:
    data_type = 'both'
    search_space = 'small'
elif machine == 3:
    data_type = 'both'
    search_space = 'full'

if machine == 4:
    data_type = 'all'
    search_space = 'fast'
elif machine == 5:
    data_type = 'all'
    search_space = 'small'
elif machine == 6:
    data_type = 'all'
    search_space = 'full'
elif machine == 7:
    data_type = 'split'
    search_space = 'fast'
elif machine == 8:
    data_type = 'split'
    search_space = 'small'
elif machine == 9:
    data_type = 'split'
    search_space = 'full'


#################################################################################
exp = 2  # can be either exp 1 or 2 for now
run_on_cloud = False
if 'hadoop' in getcwd():
    run_on_cloud2 = True
else:
    run_on_cloud2 = False
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
