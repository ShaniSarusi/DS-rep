import pickle
import boto3.session

def close_connection(co):
    csr = co.cursor()
    csr.close()
    del csr
    co.close()


def load_pickle_file_from_s3(p_region_name, p_bucket, p_file_path):
    session = boto3.session.Session(region_name=p_region_name)
    s3client = session.client('s3', config= boto3.session.Config(signature_version='s3v4'))
    response = s3client.get_object(Bucket=p_bucket, Key=p_file_path)

    body_string = response['Body'].read()
    res = pickle.loads(body_string)
    return res