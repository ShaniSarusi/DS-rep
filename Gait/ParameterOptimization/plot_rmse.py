import matplotlib.pyplot as plt
import random
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib

# plot a barchart with error bar
fig, ax = plt.subplots()

legends = ['Q1', 'Q2', 'Q3', 'Q4']
# 4 groups, generate a random list with `random.sample(range(1, 5), 4)`
groups = [	[2, 5, 1, 2],	# alg1
            [1, 2, 3, 4],	# alg2
            [3, 1, 2, 4],	# Q3
            [2, 1, 4, 3]]	# Q4

std = [	    [0.2, 0.3, 0.1, 0.2],	# Q1
            [0.1, 0.2, 0.3, 0.4],	# Q2
            [0.3, 0.1, 0.2, 0.4],	# Q3
            [0.2, 0.1, 0.4, 0.3]]	# Q4

colors = ['r', 'g', 'b', 'm']

x = range(len(groups))
list_rects = list()
for idx, group in enumerate(groups):
    left = [0.2 + i+idx*0.15 for i in x]

    # rects = plt.bar(left, group,
    #                 width=0.2,
    #                 color=colors[idx],
    #                 yerr=0.1, ecolor='k', capsize=5,
    #                 orientation='vertical')

    rects = ax.errorbar(left, group,
                    color=colors[idx],
                    yerr=std[idx], ecolor='k', capsize=5, fmt='o')

    list_rects.append(rects)


# decoration
ax.legend(list_rects, legends, loc='upper left')

# set xtick labels
list_ticklabel = ['2012', '2013', '2014', '2015']  #the tasks themselves 1,2,3,4,5...
ax.set_xticks([i+0.4 for i in x])
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
plt.show()
