import xgboost as xgb
import boto3

with open("access_s3") as f:
    access = f.read().strip().split('\n')

    ACCESS_KEY = access[0]
    ACCESS_SECRET_KEY = access[1]

def load_files_from_s3():
    s3_resource = boto3.resource(
        "s3",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=ACCESS_SECRET_KEY,
        region_name="us-east-2"
    )
    bucket = s3_resource.Bucket(name='made-classifier-food-type')

    bucket.download_file('avg_claims_model.pkl', 'avg_claims_model.pkl')
    bucket.download_file('claim_counts_model.pkl', 'claim_counts_model.pkl')


def get_model():
    load_files_from_s3()
    return lambda x: 'Test'
