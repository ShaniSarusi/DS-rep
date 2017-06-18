# takes the entire downloaded raw data (with Tomer) and stores it as a file per person, sorted by time, with only
# id, ts, x, y, and z columns. It stores each file in csv format.
import pandas as pd
import numpy as np

# Read first column (Person ID), then sort, and store indices for each person ID
common_path = 'C:\Users\zwaks\Documents\Data\Raw Data Minitrial'

csv_path = common_path + '\dataRawDataMiniTrial.csv'
print('Reading all rows, only user id column')
df = pd.read_csv(csv_path, skiprows=[0], header=None, usecols=[0])
print('Sorting user ids')
df = df.sort_values([0])
ids = df[0].unique()
total_samples = df.shape[0]

patient = 0
for id in ids:
    patient = patient+1
    print('Starting patient', patient, 'out of', ids.shape[0])
    id_idx = df[df[0] == id].index.values + 1
    skip = np.setdiff1d(range(total_samples), id_idx)
    print('Reading user', patient, 'values - [id, ts, x, y, z]')
    id_idf = pd.read_csv(csv_path, skiprows=skip, header=None, usecols=[0, 1, 5, 6, 7])
    print('Sorting by ts')
    id_idf = id_idf.sort_values([0, 1])
    id_idf.columns = ['id', 'ts', 'x', 'y', 'z']
    save_path = common_path + '\\' + str(id) + '.csv'
    print('Saving')
    id_idf.to_csv(save_path, index=False)

print('Done parsing raw data minitrial csv download')
