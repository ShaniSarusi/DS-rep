import pandas as pd


# read export tool data
def read_export_tool_csv(csv_path):
    df = pd.read_csv(csv_path, skiprows=[0], header=None)
    df.user = df.iloc[1][0]
    df.columns = ['a', 'b', 'ts', 'c', 'x', 'y', 'z']
    df.drop(['a', 'b', 'c'], axis=1, inplace=True)
    df['x'] = df['x'].str.replace('{x=', '')
    df['y'] = df['y'].str.replace('y=', '')
    df['z'] = df['z'].str.replace('z=', '')
    df['z'] = df['z'].str.replace('}', '')

    # set types
    df['x'] = df['x'].astype('float')
    df['y'] = df['y'].astype('float')
    df['z'] = df['z'].astype('float')
    df['ts'] = pd.to_datetime(df.loc[:, 'ts'])

    return df


def make_df(data, col_names):
    df = pd.DataFrame(index=range(data.shape[0]), columns=col_names)
    for i in range(data.shape[0]):
        df.iloc[i] = data[i]
    return df