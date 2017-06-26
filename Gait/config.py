from os.path import join, sep

# paths *****************************************************************
# common_path = join('C:', sep, 'Users', 'zwaks', 'Documents', 'Data', 'Intel APDM March 6 2017')
data_path = join('C:', sep, 'Users', 'zwaks', 'Documents', 'Data')
common_path = join(data_path, 'APDM June 2017')
pickle_path = join(common_path, 'Pickled')
input_path = join(common_path, 'Subjects')
results_path = join(common_path, 'Results')

#parameters ********************************
sampling_rate = 128
# lhs_wrist_sensor = 377  # 1st exp values
# rhs_wrist_sensor = 638  # 1st exp values
lhs_wrist_sensor = 1589  # 2nd exp values
rhs_wrist_sensor = 1695  # 2nd exp values

# constants
radian = 57.2958

# Example how you can also use dictionaries
# truck = dict(
#     color = 'blue',
#     brand = 'ford',
# )
# city = 'new york'
# cabriolet = dict(
#     color = 'black',
#     engine = dict(
#         cylinders = 8,
#         placement = 'mid',
#     ),
#     doors = 2,
# )