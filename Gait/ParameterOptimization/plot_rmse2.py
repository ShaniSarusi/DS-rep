from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import Gait.config as c

data = join('C:\\Users\\zwaks\\Desktop\\apdm-june2017\\small_search_space', 'Summary.csv')
data = join('C:\\Users\\zwaks\\Documents\\Data\\APDM June 2017\\Results\\param_opt', 'Summary.csv')
rotate = True
data = pd.read_csv(data)


with open(join(c.pickle_path, 'task_filters'), 'rb') as fp:
    task_filters = pickle.load(fp)
tasks = ['all'] + task_filters['Task Name'].tolist()
tasks.remove('Tug')
tasks.remove('Tug')
a, b = tasks.index('Walk - Regular'), tasks.index('Walk - Fast')
tasks[b], tasks[a] = tasks[a], tasks[b]

xlabs = ['Cane' if x == 'Asymmetry - imagine you have a cane in the right hand' else x for x in tasks]
xlabs = ['No shoe' if x == 'Asymmetry - no right shoe' else x for x in xlabs]
xlabs = ['Hands on side' if x == 'Walk - both hands side' else x for x in xlabs]
xlabs = [x[6:] if 'Walk - ' in x else x for x in xlabs]

algs = data['Algorithm'].unique().tolist()

means = []
stds = []
for i in range(len(algs)):
    means_i = []
    stds_i = []
    for j in range(len(tasks)):
        vals = data[(data['Algorithm'] == algs[i]) & (data['WalkingTask'] == tasks[j])]['RMSE']
        means_i.append(vals.mean())
        stds_i.append(vals.std())
    means.append(means_i)
    stds.append(stds_i)


########################################################################################################################
# Plotting

# plot a barchart with error bar
fig, ax = plt.subplots()

legends = algs
groups = means
errors = stds

colors = ['r', 'g', 'b', 'm']

x = range(len(groups[0]))
list_rects = list()
for idx, group in enumerate(groups):
    left = [0.2 + i+idx*0.15 for i in x]

    rects = ax.errorbar(left, group, color=colors[idx], yerr=errors[idx], ecolor='k', capsize=5, fmt='o')
    list_rects.append(rects)

# decoration
ax.legend(list_rects, legends, loc='upper right')

# set xtick labels
list_ticklabel = ['2012', '2013', '2014', '2015']  #the tasks themselves 1,2,3,4,5...
list_ticklabel = xlabs
ax.set_xticks([i+0.4 for i in x])
if rotate:
    ax.set_xticklabels(list_ticklabel, rotation=45)
else:
    ax.set_xticklabels(list_ticklabel)
ax.set_xlim(0, max(x)+1-0.2)

# Hide the right and top spines
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(True)

# Only show ticks on the bottom spine
ax.xaxis.set_ticks_position('bottom')

# y label
ax.set_ylabel('RMSE or MAPE')

out_file = 'barchart_errorbar.png'
plt.savefig(out_file)
plt.tight_layout()
plt.show()
