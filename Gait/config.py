from os.path import join, sep

machine = 1  #1, 2, 3, 4
n_folds = 5
max_evals = 200
alg = 'tpe'  # Can be 'tpe' or 'random'

if machine == 1:
    search_space = 'full'
    data_type = 'all'
elif machine == 2:
    search_space = 'full'
    data_type = 'split'
elif machine == 3:
    search_space = 'small'
    data_type = 'all'
elif machine == 4:
    search_space = 'small'
    data_type = 'split'


#################################################################################
exp = 2  # can be either exp 1 or 2 for now
run_on_cloud = False
run_on_cloud2 = True
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
