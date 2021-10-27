import xgboost as xgb
import boto3

def load_files_from_s3():
    s3_resource = boto3.resource(
        "s3",
        aws_access_key_id="AKIATSAITSQ35WWPFWKJ",
        aws_secret_access_key="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",
        region_name="us-east-2"
    )
    bucket = s3_resource.Bucket(name='made-classifier-food-type')

    bucket.download_file('avg_claims_model.pkl', 'avg_claims_model.pkl')
    bucket.download_file('claim_counts_model.pkl', 'claim_counts_model.pkl')


def get_model():
    load_files_from_s3()
    model_avgclaim = xgb.XGBRegressor()
    model_avgclaim.load_model('avg_claims_model.pkl')
    model_claimcounts = xgb.XGBClassifier()
    model_claimcounts.load_model('claim_counts_model.pkl')
    return model_avgclaim, model_claimcounts
