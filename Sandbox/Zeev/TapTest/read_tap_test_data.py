from os import listdir
from os.path import isfile, join
import pandas as pd

# get data file names
data_path = join('C', 'Users', 'zwaks', 'Documents', 'Data', 'Tap test', 'Data', 'Test')
only_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# store data files in a single dataframe
frames = []
for i in range(len(only_files)):
    tmp = pd.read_csv(join(data_path, only_files[i]))
    frames.append(tmp)
df = pd.concat(frames).reset_index()
df = df.drop('index', 1)

# add unique experiment ID to dataframe
# method assumes no two experiments started on exactly the same millisecond (same timestamp)
df['ExperimentID'] = -1
df.loc[0, 'ExperimentID'] = 0
for i in range(1,df.shape[0]):
    if df.loc[i-1, 'Test Timestamp'] == df.loc[i, 'Test Timestamp']:
        df.loc[i, 'ExperimentID'] = df.loc[i-1, 'ExperimentID']
    else:
        df.loc[i, 'ExperimentID'] = df.loc[i - 1, 'ExperimentID'] + 1


# Early analyses of tap test
