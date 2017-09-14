import pickle
from os import makedirs
from os.path import join, exists

import matplotlib.pyplot as plt
import numpy as np

from Sandbox.Zeev import Gait_old as c


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


def plot_channels_and_events(id, start_time=8, time_range=5, plot_type='all_events',
                             save_dir=join(c.results_path, 'channel_plots'), show_plot_title=False, show_plot=True,
                             save=True):
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
    l_on = np.array(apdm.loc['Gait_old - Lower Limb - Initial Contact L (s)'])
    r_on = np.array(apdm.loc['Gait_old - Lower Limb - Initial Contact R (s)'])

    l_off = np.array(apdm.loc['Gait_old - Lower Limb - Toe Off L (s)'])
    r_off = np.array(apdm.loc['Gait_old - Lower Limb - Toe Off R (s)'])

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
        if plot_type == 'left_leg_events':
            for j in range(len(l_on)):
                ax1 = plt.axvline(l_on[j], color='r', ls='-', lw=2)
            for j in range(len(l_off)):
                ax2 = plt.axvline(l_off[j], color='r', ls='--', lw=2)
            if i == 0:
                plt.legend([ax1, ax2], ["Left initial contact", "Left toe off"], loc="center left",
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
            if i == 0:
                plt.legend([ax2, ax4], ["Left toe off", "Right toe off"], loc="center left",
                           bbox_to_anchor=(0.3, 1.18), numpoints=1, fontsize=14, ncol=2)
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

        # tmp
        a = [20.0, 114.0, 171.0, 236.0, 271.0, 319.0, 373.0, 415.0, 480.0, 516.0, 581.0, 621.0, 677.0, 735.0, 779.0,
             846.0, 893.0, 945.0, 994.0, 1047.0, 1115.0, 1152.0, 1224.0, 1314.0, 1380.0, 1444.0, 1486.0, 1531.0, 1586.0,
             1649.0, 1684.0, 1746.0, 1796.0, 1857.0, 1895.0, 1961.0, 2004.0, 2066.0, 2111.0, 2260.0, 2318.0, 2380.0,
             2433.0, 2481.0, 2535.0, 2589.0, 2622.0, 2678.0, 2736.0, 2786.0, 2849.0, 2907.0, 2955.0, 3016.0, 3057.0,
             3121.0, 3157.0, 3225.0, 3268.0, 3334.0, 3380.0, 3444.0, 3478.0, 3544.0, 3607.0, 3654.0, 3716.0, 3768.0,
             3811.0]

        a = [219.0, 327.0, 408.0, 494.0, 586.0, 678.0, 744.0, 839.0, 911.0, 1014.0, 1081.0, 1174.0, 1252.0, 1348.0, 1428.0, 1517.0, 1586.0, 1680.0, 1749.0, 1845.0, 1918.0, 2008.0, 2086.0, 2180.0, 2247.0, 2337.0, 2415.0, 2511.0, 2579.0, 2671.0, 2750.0, 2838.0, 2905.0, 3001.0, 3072.0, 3171.0, 3240.0, 3333.0, 3414.0, 3503.0, 3570.0, 3668.0, 3739.0, 3835.0, 3910.0, 4004.0, 4072.0, 4173.0, 4244.0, 4341.0, 4421.0, 4512.0, 4587.0, 4683.0, 4757.0, 4864.0, 4942.0, 5042.0, 5110.0, 5198.0, 5277.0, 5363.0, 5435.0, 5530.0, 5591.0, 5691.0, 5757.0, 5849.0, 5920.0, 6020.0, 6084.0, 6190.0, 6254.0, 6344.0, 6403.0, 6493.0, 6563.0, 6658.0, 6723.0, 6821.0, 6886.0, 6989.0, 7052.0, 7159.0, 7224.0, 7327.0, 7403.0, 7508.0, 7568.0, 7662.0, 7724.0, 7826.0, 7891.0, 7989.0, 8057.0, 8160.0, 8224.0, 8321.0, 8387.0, 8483.0, 8549.0, 8645.0, 8707.0, 8798.0, 8877.0, 8972.0, 9036.0, 9138.0, 9213.0, 9311.0, 9379.0, 9468.0, 9547.0, 9645.0, 9714.0, 9821.0, 9948.0]
        #a = [53.0, 137.0, 241.0, 338.0, 412.0, 492.0, 561.0, 632.0, 698.0, 774.0, 835.0, 917.0, 974.0, 1060.0, 1119.0, 1203.0, 1272.0, 1343.0, 1397.0, 1479.0, 1543.0, 1619.0, 1680.0, 1756.0, 1812.0, 1885.0, 1947.0, 2015.0, 2075.0, 2143.0, 2201.0, 2275.0, 2331.0, 2403.0, 2457.0, 2532.0, 2587.0, 2665.0, 2720.0, 2798.0, 2859.0, 2935.0, 2984.0, 3061.0, 3125.0, 3195.0, 3248.0, 3327.0, 3377.0, 3459.0, 3501.0, 3588.0, 3643.0, 3718.0, 3768.0, 3850.0, 3903.0, 3979.0, 4025.0, 4085.0, 4160.0, 4240.0, 4295.0, 4369.0, 4420.0, 4495.0, 4550.0, 4625.0, 4689.0, 4757.0, 4808.0, 4881.0, 4936.0, 5010.0, 5064.0, 5143.0, 5191.0, 5270.0, 5331.0, 5400.0, 5479.0, 5528.0, 5578.0, 5659.0, 5710.0, 5789.0, 5839.0, 5919.0, 5969.0, 6047.0, 6102.0, 6178.0, 6231.0, 6309.0, 6362.0, 6442.0, 6505.0, 6578.0, 6659.0, 6710.0, 6755.0, 6842.0, 6901.0, 6973.0, 7022.0, 7103.0, 7156.0, 7235.0, 7289.0, 7376.0, 7468.0, 7565.0, 7653.0]


        b = [i / 128.0 for i in a]
        for j in range(len(b)):
            ax5 = plt.axvline(b[j], color='g', ls='-', lw=2)
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
    id = 170
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
        plot_channels_and_events(id, start_time=42, time_range=8, plot_type=plot_type, save=False, show_plot=True)
