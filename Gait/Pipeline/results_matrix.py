import pickle
import pandas as pd
from os.path import join
import Gait.config as c


def create_result_matrix():
    with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp: sample = pickle.load(fp)
    with open(join(c.pickle_path, 'features_steps'), 'rb') as fp: step_features = pickle.load(fp)
    # with open(join(common_input, 'features_armswing'), 'rb') as fp: arm_swing_features = pickle.load(fp)

    # connect DataFrame
    step_features = step_features.drop('step_durations', axis=1)

    # df_results = pd.concat([step_features, arm_swing_features], axis=1)
    df_results = step_features
    df_sample_and_results = pd.concat([sample, df_results], axis=1)

    # save csvs
    df_sample_and_results.to_csv(join(c.results_path, 'Results.csv'))


if __name__ == '__main__':
    create_result_matrix()
