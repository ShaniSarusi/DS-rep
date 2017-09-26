import pickle
from os import makedirs
from os.path import join, exists, sep
import matplotlib.pyplot as plt
import numpy as np
import Gait.Resources.config as c
import pandas as pd
from Utils.DataHandling.data_processing import string_to_int_list


def do_everything(start_time=8, time_range=5, save_dir=join(c.results_path, 'channel_plots')):
    with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
        sample = pickle.load(fp)
    ids = sample[sample['StepCount'].notnull()].index.tolist()
    for id in ids:
        save_dir_id = join(save_dir, str(id))
        if not exists(save_dir_id):
            makedirs(save_dir_id)
        plot_channels_all_options(id, start_time, time_range, save_dir_id)


def plot_channels_all_options(id, start_time=8, time_range=5, save_dir=(c.results_path, 'channel_plots')):
    plot_type = ['left_leg_events', 'initial_contacts', 'toes_off', 'all_events']
    for p_type in plot_type:
        plot_channels_and_events(id, start_time, time_range, plot_type=p_type, save_dir=save_dir, show_plot=False)


def plot_channels_and_events(id, file=None, alg=None, start_time=8, time_range=5, plot_type='all_events',
                             save_dir=join(c.results_path, 'channel_plots'), show_plot_title=False, show_plot=True,
                             save=True):
    # Set signal channels
    with open(join(c.pickle_path, 'acc'), 'rb') as fp:
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
    l_on = np.array(apdm.loc['Gait - Lower Limb - Initial Contact L (s)'])
    r_on = np.array(apdm.loc['Gait - Lower Limb - Initial Contact R (s)'])

    l_off = np.array(apdm.loc['Gait - Lower Limb - Toe Off L (s)'])
    r_off = np.array(apdm.loc['Gait - Lower Limb - Toe Off R (s)'])

    # No APDM events (or not enough)
    if np.any([np.isnan(l_on), np.isnan(r_on), np.isnan(l_off), np.isnan(r_off)]):
        return

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
        # add alg_idx
        data = pd.read_csv(input_file)
        data_id = data[data['SampleId'] == id].index[0]
        idx = string_to_int_list(data['idx_' + alg][data_id])
        alg_ts = [(acc[id]['lhs']['ts'][i] - acc[id]['lhs']['ts'].iloc[0]) / np.timedelta64(1, 's') for i in idx]

        if plot_type == 'left_leg_events':
            for j in range(len(l_on)):
                ax1 = plt.axvline(l_on[j], color='r', ls='-', lw=2)
            for j in range(len(l_off)):
                ax2 = plt.axvline(l_off[j], color='r', ls='--', lw=2)
            if i == 0:
                plt.legend([ax1, ax2, ax5], ["Left initial contact", "Left toe off"], loc="center left",
                           bbox_to_anchor=(0.3, 1.18), numpoints=1, fontsize=14, ncol=2)
        elif plot_type == 'initial_contacts':
            for j in range(len(l_on)):
                ax3 = plt.axvline(l_on[j], color='r', ls='-', lw=2)
            for j in range(len(r_on)):
                ax4 = plt.axvline(r_on[j], color='k', ls='-', lw=2)
            if i == 0:
                plt.legend([ax3, ax4], ["Left initial contact", "Right initial contact"], loc="center left",
                           bbox_to_anchor=(0.3, 1.18), numpoints=1, fontsize=14, ncol=2)
        elif plot_type == 'toes_off':
            for j in range(len(l_off)):
                ax2 = plt.axvline(l_off[j], color='r', ls='--', lw=2)
            for j in range(len(r_off)):
                ax4 = plt.axvline(r_off[j], color='k', ls='--', lw=2)
            for j in range(len(alg_ts)):
                ax5 = plt.axvline(alg_ts[j], color='g', ls='-', lw=2)
            if i == 0:
                plt.legend([ax2, ax4, ax5], ["Left toe off", "Right toe off", "Algorithm"], loc="center left",
                           bbox_to_anchor=(0.2, 1.18), numpoints=1, fontsize=14, ncol=3)
        else:  # 'all_events'
            for j in range(len(l_on)):
                ax1 = plt.axvline(l_on[j], color='r', ls='-', lw=2)
            for j in range(len(l_off)):
                ax2 = plt.axvline(l_off[j], color='r', ls='--', lw=2)
            for j in range(len(r_on)):
                ax3 = plt.axvline(r_on[j], color='k', ls='-', lw=2)
            for j in range(len(r_off)):
                ax4 = plt.axvline(r_off[j], color='k', ls='--', lw=2)
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
        save_path = join(save_dir, save_name)
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()


if __name__ == '__main__':
    do_and_save_everything = False
    do_all_plots = False
    id = 145  #20, 204, 214
    id = 193
    #id = 207

    if do_and_save_everything:
        do_everything(start_time=8, time_range=5)
    elif do_all_plots:
        plot_channels_all_options(id, start_time=8, time_range=5)
    else:
        #plot_type = 'left_leg_events'
        # plot_type = 'initial_contacts'
        plot_type = 'toes_off'
        #plot_type = 'all_events'
        save_dir = join('C:', sep, 'Users', 'zwaks', 'Desktop', 'GaitPaper')
        input_file = join(save_dir, 'aa_param1small_0.1k_sc_v2', 'gait_measures.csv')
        plot_channels_and_events(id, file=input_file, alg='fusion_high_level_union',
                                 start_time=2, time_range=18, plot_type=plot_type, save=False, show_plot=True)
