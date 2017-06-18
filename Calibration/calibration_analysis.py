import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SMALL_SIZE = 10
MEDIUM_SIZE = 16
BIGGER_SIZE = 24
# combine single minute intervals

common_path = 'C:\Users\zwaks\Documents\Data\Raw Data Minitrial\By user per minute\\'
f = common_path + 'all_six_min intervals.csv'
df = pd.read_csv(f)
df2 = pd.read_csv(common_path + 'all_combined.csv')
x = df['Xstd']
y = df['Ystd']
z = df['Zstd']
dur = df['Duration']



n, bins, patches = plt.hist(x.dropna(), 50, normed=1, facecolor='green', alpha=0.75)
plt.xlabel('X-axis: Standard devation of x-values in each 1min interval')
plt.ylabel('Probability')

# zoom in
p = [142294, 142295, 142301, 142305, 142321, 142322]
x = df[df['PatientID'] == p[0]]['Xstd']


fig = plt.figure()
n, bins, patches = plt.hist(x.dropna(), 2000, normed=1, facecolor='green', alpha=0.75)
plt.xlabel('Axis standard devation', fontsize=BIGGER_SIZE)
plt.ylabel('Probability', fontsize=BIGGER_SIZE)
plt.xticks(size=MEDIUM_SIZE)
plt.yticks(size=MEDIUM_SIZE)
plt.axis([0, 50, 0, 0.3])
fig.savefig(common_path + 'xaxis.jpg')

most_common_val= bins[np.argmax(n)]
most_common_val

# duration histogram
fig = plt.figure()
n, bins, patches = plt.hist(dur.dropna(), 2000, normed=1, facecolor='green', alpha=0.75)
plt.xlabel('Axis standard devation', fontsize=BIGGER_SIZE)
plt.ylabel('Probability', fontsize=BIGGER_SIZE)
plt.xticks(size=MEDIUM_SIZE)
plt.yticks(size=MEDIUM_SIZE)
plt.axis([0, 50, 0, 0.1])
fig.savefig(common_path + 'duration.jpg')


#3d scatter
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#loop over to show color per patient.
x = df2[df2['Type'] == 'Static']['Xmean']
y = df2[df2['Type'] == 'Static']['Ymean']
z = df2[df2['Type'] == 'Static']['Zmean']

ax.scatter(x, y, z)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

xb = x.dropna().as_matrix()
yb = y.dropna().as_matrix()
zb = z.dropna().as_matrix()

data = [xb, yb, zb]

fig = plt.figure()
i = 0
for y in [xb, yb , zb]:
    i += 1
    # Add some random "jitter" to the x-axis
    x = np.random.normal(i, 0.04, size=len(y))
    plt.plot(x, y, 'r.', alpha=0.2)
x = np.array([0,1,2,3])
my_xticks = ['','X','Y','Z']
plt.xticks(x, my_xticks)
plt.axis([0.5, 3.5, -1200, 1200])
fig.savefig(common_path + 'xyz box.jpg')


# plot per 24hrs # of activity types
t = pd.to_datetime(df2.loc[:, 'TSstart'])
for i in range(df2.shape[0]):
    df2.loc[i, 'st'] = t[i].hour + float(t[i].minute)/60
    df2.loc[i, 'en'] = df2.loc[i, 'st'] + df2.loc[i, 'Duration']

# static = df[''.groupby('time').size()

x = []
resolution = 5 #minutes
for i in range(24):
    for j in range(60):
        t = i + float(j)/ (60/resolution)
        x.append(t)

static = np.zeros(len(x))
activity = np.zeros(len(x))
no_broadcast = np.zeros(len(x))
for i in range(len(x)):
    print(i)
    for j in range(df2.shape[0]):
        if df2.loc[j, 'st'] <= x[i] < df2.loc[j, 'en']:
            if df2.loc[j, 'Type'] == 'Static':
                static[i] += 1
            if df2.loc[j, 'Type'] == 'Activity':
                activity[i] += 1
            if df2.loc[j, 'Type'] == 'No broadcast':
                no_broadcast[i] += 1

plt.plot(x, static, 'r')
plt.plot(x, activity, 'g')
plt.plot(x, no_broadcast, 'b')


