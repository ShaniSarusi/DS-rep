from os import listdir
from os.path import isfile, join
import pandas as pd
from datetime import timedelta
import re
from Calibration import AutomaticCalibration
from os.path import join

# get data file names
data_path = 'C:\Users\zwaks\Documents\Data\Raw Data Minitrial\By user'
patient_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

win_size = timedelta(minutes=1)
out_path = 'C:\Users\zwaks\Documents\Data\Raw Data Minitrial\By user per minute'
# for i in range(4, 5):  # only file 4, the smallest one
for i in range(len(patient_files)):
    print('In person ' + str(i + 1) + ' of ' + str(len(patient_files)))
    file_path = join(data_path, patient_files[i])
    df = pd.read_csv(file_path)
    df['ts'] = pd.to_datetime(df.loc[:, 'ts'])

    patient_id = df.loc[0,'id']
    df.drop(['id'], axis=1, inplace=True)
    cal = AutomaticCalibration(df, static_win_duration=win_size, std_thresh=30)
    # Create a data-frame with data about the 1-minute windows in the data, then  write it to CSV:
    cal.calc_windows(verbose=True)
    out = out_path + '\\' + str(patient_id) + '_all.csv'
    cal.windows.to_csv(out, index=False)

# find & combine static windows, & save file per user & one joint file
patient_files = [f for f in listdir(out_path) if isfile(join(out_path, f))]

for f in patient_files:
    file_path = out_path + '\\' + f
    df = pd.read_csv(file_path)
    # find window types
    x_thresh = 6; y_thresh = 6; z_thresh = 6; min_norm = 500; max_norm = 1500
    types = []
    for i in range(df.shape[0]):
        if df.loc[i, 'NumSamples'] == 0:
            w = 'No broadcast'
        elif df.loc[i, 'Xstd'] > x_thresh:
            w = 'Activity'
        elif df.loc[i, 'Ystd'] > y_thresh:
            w = 'Activity'
        elif df.loc[i, 'Zstd'] > z_thresh:
            w = 'Activity'
        elif df.loc[i, 'Nmean'] < min_norm:
            w = 'Activity'
        elif df.loc[i, 'Nmean'] > max_norm:
            w = 'Activity'
        else:
            w = 'Static'
        types.append(w)

    # get window type ranges
    i = 0
    ran = []
    while i < df.shape[0]:
        for j in range(i + 1, df.shape[0] + 1):
            if j == df.shape[0]:
                break
            if types[i] != types[j]:
                break
        ran.append(range(i, j))
        i = j

    # combine windows
    cols = df.columns
    cols = cols.insert(2, "Type")
    cols = cols.insert(0, "UserID")
    windows = pd.DataFrame(index=range(len(ran)), columns=cols)
    userid = re.sub("_all.csv", "", f)
    for i in range(len(ran)):
        # calculations and append
        sampleid = i + 1
        ts = df['TSstart'][ran[i][0]]
        ty = types[ran[i][0]]
        dur = len(ran[i])
        num_samples = df['NumSamples'][ran[i]].sum()
        xstd = df['Xstd'][ran[i]].mean()
        ystd = df['Ystd'][ran[i]].mean()
        zstd = df['Zstd'][ran[i]].mean()
        nstd = df['Nstd'][ran[i]].mean()
        xm = df['Xmean'][ran[i]].mean()
        ym = df['Ymean'][ran[i]].mean()
        zm = df['Zmean'][ran[i]].mean()
        nm = df['Nmean'][ran[i]].mean()
        windows.iloc[i] = [userid, sampleid, ts, ty, dur, num_samples, xstd, ystd, zstd, nstd, xm, ym, zm, nm]
    out = re.sub("all.csv", "combined_win.csv", file_path)
    windows.to_csv(out, index=False)

