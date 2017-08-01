"""
This module provides various functions to handle connections.
"""
import pickle
import boto3.session
import time


def close_connection(co):
    """
    Close the database connection that is passed into the function.

    Input
    co (object of type (Connection?)): The input connection
    """
    csr = co.cursor()
    csr.close()
    del csr
    co.close()


def load_pickle_file_from_s3(p_region_name, p_bucket, p_file_path):
    """
    Load the pickle file specified by p_file_path in s3 and return it.

    Input:
    p_region_name (string): The AWS region name. Example: us-west-2
    p_bucket (string):  The AWS bucket where the file resides. Example: 'intel-health-analytics'
    p_file_path (string): The file path in s3, starting after the bucket.

    Output
    out1 (often dictionary): The data that is contained in the pickle file specified by the input
    """
    session = boto3.session.Session(region_name=p_region_name)
    s3client = session.client('s3', config= boto3.session.Config(signature_version='s3v4'))
    response = s3client.get_object(Bucket=p_bucket, Key=p_file_path)

    body_string = response['Body'].read()
    res = pickle.loads(body_string)
    return res


def save_pickle_file_to_s3(p_region_name, p_bucket, p_local_file):
    """
    Save the pickle file specified by p_local_file in the p_bucket in s3.

    Input:
    p_region_name (string): The AWS region name. Example: us-west-2
    p_bucket (string):  The AWS bucket where the file will be saved. Example: 'intel-health-analytics'
    p_local_file (string): The local path of the pickled file.
    """
    session = boto3.session.Session(region_name=p_region_name)
    s3client = session.client('s3', config= boto3.session.Config(signature_version='s3v4'))

    time_string = time.strftime("%Y%m%d-%H%M%S")
    s3client.upload_file(p_local_file, p_bucket, 'param_opt_results-' + time_string)

