import Gait.Pipeline.read_data as p1
import Gait.Pipeline.features_generic as p2
# import Gait.Pipeline.step_fts as p3
import Gait.Pipeline.features_arms as p4
import Gait.Pipeline.results_matrix as p5
import Gait.config as c
from os.path import join
import pickle


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

# Store APDM results
apdm_files = p1.read_apdm_result_files()
apdm_measures, apdm_events = p1.extract_apdm_results(apdm_files, p_len_raw_data=len(acc))
with open(join(c.pickle_path, 'apdm_measures'), 'wb') as fp: pickle.dump(apdm_measures, fp)
with open(join(c.pickle_path, 'apdm_events'), 'wb') as fp: pickle.dump(apdm_events, fp)


# Extract features
p2.extract_generic_features()
# p3.extract_step_features()
# p4.extract_arm_swing_features()

# Create result summary
# p5.create_result_matrix()