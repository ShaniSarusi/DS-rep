from os.path import join
import pickle
import Gait.Resources.config as c
import matplotlib.pyplot as plt
import numpy as np


def plot_channels_all_options(id, start_time=8, time_range=5):
    plot_type = ['left_leg_events', 'initial_contacts', 'toes_off', 'all_events']
    for p_type in plot_type:
        plot_channels_and_events(id, start_time, time_range, plot_type=p_type, show_plot=False)


def plot_channels_and_events(id, start_time=8, time_range=5, plot_type='all_events', show_plot_title=False,
                             show_plot=True, save=True):
    # Set signal channels
    with open(join(c.pickle_path, 'acc_leg_lhs'), 'rb') as fp:
        acc = pickle.load(fp)
    lhs = acc[id]['lhs']['n']
    rhs = acc[id]['rhs']['n']
    leg = acc[id]['leg_lhs']['n']

    lhs = (lhs - min(lhs))/(max(lhs) - min(lhs))
    rhs = (rhs - min(rhs))/(max(rhs) - min(rhs))
    leg = (leg - min(leg))/(max(leg) - min(leg))

    # Set APDM event times
    with open(join(c.pickle_path, 'apdm_events'), 'rb') as fp:
        apdm_events = pickle.load(fp)
    apdm = apdm_events.loc[id]
    lic = np.array(apdm.loc['Gait - Lower Limb - Initial Contact L (s)'])
    lfc = np.array(apdm.loc['Gait - Lower Limb - Toe Off L (s)'])
    ric = np.array(apdm.loc['Gait - Lower Limb - Initial Contact R (s)'])
    rfc = np.array(apdm.loc['Gait - Lower Limb - Toe Off R (s)'])

    # Set plotting parameters
    x_tick_spacing = 1
    x_values = np.arange(len(lhs))/c.sampling_rate
    plt.figure(figsize=(12, 6))
    channels = [leg, lhs, rhs]
    num_plots = len(channels)
    y_titles = ['Left leg', 'Left wrist', 'Right wrist']
    for i in range(num_plots):
        plt.subplot(num_plots, 1, i + 1)
        plt.plot(x_values, channels[i])
        x_max = start_time + time_range
        plt.xlim([start_time, x_max])
        plt.xticks(np.arange(start_time, x_max + x_tick_spacing, x_tick_spacing))
        plt.ylabel(y_titles[i], fontsize=14)
        plt.yticks([])
        if plot_type == 'left_leg_events':
            for j in range(len(lic)):
                ax1 = plt.axvline(lic[j], color='k', ls='-.', lw=2)
            for j in range(len(lfc)):
                ax2 = plt.axvline(lfc[j], color='r', ls='-.', lw=2)
            if i == 0:
                plt.legend([ax1, ax2], ["Left initial contact", "Left toe off"], loc="center left",
                           bbox_to_anchor=(0.3, 1.18), numpoints=1, fontsize=14, ncol=2)
        elif plot_type == 'initial_contacts':
            for j in range(len(lic)):
                ax1 = plt.axvline(lic[j], color='k', ls='-.', lw=2)
            for j in range(len(ric)):
                ax2 = plt.axvline(ric[j], color='r', ls='-.', lw=2)
            if i == 0:
                plt.legend([ax1, ax2], ["Left initial contact", "Right initial contact"], loc="center left",
                           bbox_to_anchor=(0.3, 1.18), numpoints=1, fontsize=14, ncol=2)
        elif plot_type == 'toes_off':
            for j in range(len(lfc)):
                ax1 = plt.axvline(lfc[j], color='k', ls='-.', lw=2)
            for j in range(len(rfc)):
                ax2 = plt.axvline(rfc[j], color='r', ls='-.', lw=2)
            if i == 0:
                plt.legend([ax1, ax2], ["Left toe off", "Right toe off"], loc="center left",
                           bbox_to_anchor=(0.3, 1.18), numpoints=1, fontsize=14, ncol=2)
        else:  # 'all_events'
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
                                                  "Right toe off"], loc="center left", bbox_to_anchor=(0.25, 1.24),
                           numpoints=1, fontsize=13, ncol=2)
    plt.xlabel('Seconds', fontsize=16)
    if show_plot_title:
        plt.suptitle('Initial contact and toe off (legs vs wrist)', fontsize=18)

    # Save and show
    if save:
        save_name = 'id' + str(id) + '_' + str(start_time) + 'to' + str(start_time+time_range) + 's_' + plot_type + '.png'
        save_path = join(c.results_path, 'channel_plots', save_name)
        plt.savefig(save_path)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    do_all_plots = True
    id = 134  #20, 204, 214

    if do_all_plots:
        plot_channels_all_options(id, start_time=8, time_range=5)
    else:
        #plot_type = 'left_leg_events'
        plot_type = 'initial_contacts'
        #plot_type = 'toes_off'
        #plot_type = 'all_events'
        plot_channels_and_events(id, start_time=8, time_range=5, plot_type=plot_type)
