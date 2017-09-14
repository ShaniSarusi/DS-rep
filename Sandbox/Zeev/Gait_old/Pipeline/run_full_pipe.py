import pickle
from os.path import join

import Gait_old.Pipeline.extract_features as p2
# import Gait_old.Pipeline.step_fts as p3
import Gait_old.Resources.config as c

import Sandbox.Zeev.Gait_old.Pipeline.read_data as p1

# Store metadata
p1.pickle_metadata()

# Read file names
raw_data_input_file_names = p1.read_input_files_names()
apdm_files = p1.read_apdm_result_files()

# Read and process raw signal data
acc, bar, gyr, mag, temp, time = p1.extract_sensor_data(raw_data_input_file_names)
acc = p1.add_norm(acc)
gyr = p1.add_norm(gyr)
mag = p1.add_norm(mag)
acc, bar, gyr, mag, temp, time = p1.fix_data_types(acc, bar, gyr, mag, temp, time)
acc, bar, gyr, mag, temp = p1.add_ts_to_sensor_data(acc, bar, gyr, mag, temp, time)

# Truncate signal data and labels
p1.metadata_truncate_start_label(time)
acc, bar, gyr, mag, temp, time = p1.truncate_start_sensor_data(acc, bar, gyr, mag, temp, time)
p1.pickle_sensor_data(acc, bar, gyr, mag, temp, time)

# Store APDM results
apdm_measures, apdm_events = p1.extract_apdm_results(apdm_files, raw_data_input_file_names, p_len_raw_data=len(acc))
with open(join(c.pickle_path, 'apdm_measures'), 'wb') as fp: pickle.dump(apdm_measures, fp)
with open(join(c.pickle_path, 'apdm_events'), 'wb') as fp: pickle.dump(apdm_events, fp)


# Extract features
p2.extract_generic_features()
# p2.extract_arm_swing_features()

# Create result summary
# p1.create_result_matrix()
