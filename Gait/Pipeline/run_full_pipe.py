import Gait.Pipeline.read_data as p1
import Gait.Pipeline.features_generic as p2
# import Gait.Pipeline.step_fts as p3
import Gait.Pipeline.features_arms as p4
import Gait.Pipeline.results_matrix as p5
from os.path import join

# data processing/pre-processing
p1.pickle_metadata()
input_files = p1.read_input_files_names()
acc, bar, gyr, mag, temp, time = p1.extract_sensor_data(input_files)
acc = p1.add_norm(acc)
gyr = p1.add_norm(gyr)
mag = p1.add_norm(mag)
acc, bar, gyr, mag, temp, time = p1.fix_data_types(acc, bar, gyr, mag, temp, time)
acc, bar, gyr, mag, temp = p1.add_ts_to_sensor_data(acc, bar, gyr, mag, temp, time)

# p1.metadata_truncate_start_label(time)
acc, bar, gyr, mag, temp, time = p1.truncate_start_sensor_data(acc, bar, gyr, mag, temp, time)
p1.pickle_sensor_data(acc, bar, gyr, mag, temp, time)

# extract features
p2.extract_generic_features()
# p3.extract_step_features()
#p4.extract_arm_swing_features()

# create result summary
#p5.create_result_matrix()