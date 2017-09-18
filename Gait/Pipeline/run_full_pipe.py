import pickle
from os.path import join
import Gait.Resources.config as c
from Gait.Pipeline.read_data import pickle_metadata, read_input_files_names, read_apdm_result_files, extract_sensor_data, add_norm, fix_data_types, \
    add_ts_to_sensor_data, metadata_truncate_start_label, truncate_start_sensor_data, pickle_sensor_data, extract_apdm_results

# Store metadata
pickle_metadata()

# Read file names
raw_data_input_file_names = read_input_files_names()
apdm_files = read_apdm_result_files()

# Read and process raw signal data
acc, bar, gyr, mag, temp, time = extract_sensor_data(raw_data_input_file_names)
acc = add_norm(acc)
gyr = add_norm(gyr)
mag = add_norm(mag)
acc, bar, gyr, mag, temp, time = fix_data_types(acc, bar, gyr, mag, temp, time)
acc, bar, gyr, mag, temp = add_ts_to_sensor_data(acc, bar, gyr, mag, temp, time)

# Truncate signal data and labels
metadata_truncate_start_label(time)
acc, bar, gyr, mag, temp, time = truncate_start_sensor_data(acc, bar, gyr, mag, temp, time)
pickle_sensor_data(acc, bar, gyr, mag, temp, time)

# Store APDM results
apdm_measures, apdm_events = extract_apdm_results(apdm_files, raw_data_input_file_names, p_len_raw_data=len(acc))
with open(join(c.pickle_path, 'apdm_measures'), 'wb') as fp: pickle.dump(apdm_measures, fp)
with open(join(c.pickle_path, 'apdm_events'), 'wb') as fp: pickle.dump(apdm_events, fp)
