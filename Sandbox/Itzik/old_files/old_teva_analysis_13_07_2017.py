# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 10:43:23 2017

@author: imazeh
"""


def plot_statistics_distribution_w_min_max(df, assessment_id, user_id, target_variable,
                                           statistics=statistics+['min', 'max'], min_val=min_val, max_val=max_val):
    print('\nNumber of assessments per user in this dataset:')
    print(df[[user_id, assessment_id]].groupby(user_id)[assessment_id].nunique(), '\n')
    grouped_reports = df[[assessment_id, user_id, target_variable]].groupby(assessment_id, as_index=False).agg('mean')
    reports = grouped_reports[target_variable].astype(int)
    # Count report appearances for each possible value (even if didn't appear):
    counts = []
    for x in c.report_values:
        x_count = len([r for r in reports if r==x])
        counts.append(x_count)
    # Plot:    
    fig, ax = plt.subplots()
    ax.bar(c.report_values, counts, width=0.4, align='center')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('Histogram of %s across all records:' % (target_variable))
    ax.set_xlabel('Reported Chorea')
    ax.set_ylabel('Count')
    plt.show()

    
    # Aggregate (mean, std, median) per patient, and plot all into one graph:
    print('\n')
    per_user_stats = grouped_reports[[user_id, target_variable]].groupby(user_id, as_index=False).agg(statistics)
    per_user_stats.columns = per_user_stats.columns.droplevel(0)
    per_user_stats.reset_index(inplace=True)
    
    n_users = per_user_stats.shape[0]
    x_locs = np.arange(1, n_users+1)
    width = 0.18
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    means = per_user_stats['mean']
    stds = per_user_stats['std']
    medians = per_user_stats['median']
    mins = per_user_stats['min']
    maxs = per_user_stats['max']

    rects1 = ax.bar(x_locs, means, width, color='g', yerr=stds)
    rects2 = ax.bar(x_locs+width, medians, width, color='b')
    rects3 = ax.bar(x_locs+(2*width), mins, width, color='r')
    rects4 = ax.bar(x_locs+(3*width), maxs, width, color='y')
    
    ax.set_title('Summary statistics per patient')
    ax.set_ylabel('Reported Chorea')
    ax.set_xlabel('Patient ID')
    ax.set_xticks(x_locs+(1.5*width))
    ax.set_xticklabels(per_user_stats['user_id'].astype(str))
    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('Mean', 'Median', 'Min', 'Max'))
    plt.show()
    
    
    # Calculate the fraction of 'positive' reports per patient, and plot:
    print('\n')
    grouped_reports['binary_report'] = grouped_reports[target_variable]>=1
    binary_avg = grouped_reports[[user_id, 'binary_report']].groupby(user_id, as_index=False).agg('mean')
    binary_avg['user_id'] = binary_avg['user_id'].astype(str)

    plt.figure(3)
    plt.bar(x_locs, binary_avg['binary_report'])
    plt.xticks(x_locs, binary_avg['user_id'])
    plt.title('Fraction of positive (>1) reports per patient')
    plt.xlabel('Patient ID')
    plt.ylabel('Fraction of positive reports')
    plt.show()
    
    return


def old_plot_statistics_distribution(df, assessment_identifier, grouping_identifier, target_variable,
                                 statistics, min_val=0, max_val=4):
    print('\nNumber of assessments per user in this dataset:')
    print(df[[grouping_identifier, assessment_identifier]].groupby(grouping_identifier)[assessment_identifier].nunique())
    hist_range = (min_val, max_val)
    ticks = range(min_val, max_val+1)
    grouped_reports = df[[assessment_identifier, target_variable]].groupby(assessment_identifier).agg('mean')
    print('\nHistogram of %s across all records:' % (target_variable))
    plt.hist(grouped_reports[target_variable], range=hist_range)
    plt.show()
    for stat in statistics:
        agg_df = df[[grouping_identifier, target_variable]].groupby(grouping_identifier, as_index=False).agg(stat)
        if stat==np.std:
            stat = 'standard deviation'
        agg_df.columns = [grouping_identifier, stat]
        stat_vals = agg_df[stat]
        print('\n', target_variable, 'per-patient %s distribution:' % (stat))
        if stat!=np.std:
            hist_range = hist_range
        plt.hist(stat_vals, range=hist_range)
        plt.show()
    df['binary_report'] = df[target_variable]>=1
    binary_avg = df[[grouping_identifier, 'binary_report']].groupby(grouping_identifier, as_index=False).agg('mean')
    print('\n Per-patient fraction of positive reports distribution:')
    plt.hist(binary_avg['binary_report'])
    plt.show()