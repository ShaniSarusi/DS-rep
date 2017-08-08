from os.path import join
import pickle
import Gait.Resources.config as c
import matplotlib.pyplot as plt
import numpy as np

with open(join(c.pickle_path, 'acc_leg_lhs'), 'rb') as fp:
    acc = pickle.load(fp)
with open(join(c.pickle_path, 'apdm_events'), 'rb') as fp:
    apdm_events = pickle.load(fp)

norm = 'range'
norm = 'range_zero'
id = 134
id = 20
id = 204
id = 214

#
apdm = apdm_events.loc[id]
lic = np.array(apdm.loc['Gait - Lower Limb - Initial Contact L (s)'])
lfc = np.array(apdm.loc['Gait - Lower Limb - Toe Off L (s)'])
ric = np.array(apdm.loc['Gait - Lower Limb - Initial Contact R (s)'])
rfc = np.array(apdm.loc['Gait - Lower Limb - Toe Off R (s)'])

lhs = acc[id]['lhs']['n']
rhs = acc[id]['rhs']['n']
leg = acc[id]['leg_lhs']['n']

# lhs = lhs/(max(lhs) - min(lhs))
# rhs = rhs/(max(rhs) - min(rhs))
# leg = leg/(max(leg) - min(leg))

lhs = (lhs - min(lhs))/(max(lhs) - min(lhs))
rhs = (rhs - min(rhs))/(max(rhs) - min(rhs))
leg = (leg - min(leg))/(max(leg) - min(leg))


plot_type = 'left_only'
plot_type = 'initial_only'
plot_type = 'off_only'
xmin = 8
xrange = 5
tick_spacing = 1
x = np.arange(len(lhs))/128
num_plots = 3
plt.figure()
sig = [leg, lhs, rhs]
ytitle = ['Left leg', 'Left wrist', 'Right wrist']
for i in range(num_plots):
    plt.subplot(num_plots, 1, i + 1)
    plt.plot(x, sig[i])
    xmax = xmin + xrange
    plt.xlim([xmin, xmax])
    plt.xticks(np.arange(xmin, xmax + tick_spacing, tick_spacing))
    plt.ylabel(ytitle[i], fontsize=14)
    plt.yticks([])
    if plot_type == 'left_only':
        for j in range(len(lic)):
            ax1 = plt.axvline(lic[j], color='k', ls='-.', lw=2)
        for j in range(len(lfc)):
            ax2 = plt.axvline(lfc[j], color='r', ls='-.', lw=2)
        if i == 0:
            plt.legend([ax1, ax2], ["Left initial contact", "Left toe off"], loc="center left",
                       bbox_to_anchor=(0.7, 1.18), numpoints=1, fontsize=14, ncol=2)
    elif plot_type == 'initial_only':
        for j in range(len(lic)):
            ax1 = plt.axvline(lic[j], color='k', ls='-.', lw=2)
        for j in range(len(ric)):
            ax2 = plt.axvline(ric[j], color='r', ls='-.', lw=2)
        if i == 0:
            plt.legend([ax1, ax2], ["Left initial contact", "Right initial contact"], loc="center left",
                       bbox_to_anchor=(0.7, 1.18), numpoints=1, fontsize=14, ncol=2)
    elif plot_type == 'off_only':
        for j in range(len(lfc)):
            ax1 = plt.axvline(lfc[j], color='k', ls='-.', lw=2)
        for j in range(len(rfc)):
            ax2 = plt.axvline(rfc[j], color='r', ls='-.', lw=2)
        if i == 0:
            plt.legend([ax1, ax2], ["Left toe off", "Right toe off"], loc="center left",
                       bbox_to_anchor=(0.7, 1.18), numpoints=1, fontsize=14, ncol=2)
    else:
        for j in range(len(lic)):
            ax1 = plt.axvline(lic[j], color='k', ls='-.', lw=2)
        for j in range(len(lfc)):
            ax2 = plt.axvline(lfc[j], color='r', ls='-.', lw=2)
        for j in range(len(ric)):
            ax3 = plt.axvline(ric[j], color='b', ls='-.', lw=2)
        for j in range(len(rfc)):
            ax4 = plt.axvline(rfc[j], color='g', ls='-.', lw=2)
        if i == 0:
            plt.legend([ax1, ax2, ax3, ax4], ["Left initial contact", "Left toe off", 'Right initial contact',
                                              "Right toe off"], loc="center left", bbox_to_anchor=(0.7, 1.18),
                       numpoints=1, fontsize=14, ncol=2)
plt.xlabel('Seconds', fontsize=16)
plt.suptitle('Initial contact and toe off (legs vs wrist)', fontsize=18)

