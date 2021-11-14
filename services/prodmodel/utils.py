import logging
from logging.handlers import RotatingFileHandler
import sys
from boto3 import resource

def load_files_from_s3(s3_path, path, filepath):
    with open("access_s3") as f:
        access = f.read().strip().split('\n')

        ACCESS_KEY = access[0]
        ACCESS_SECRET_KEY = access[1]

    s3_resource = resource(
        "s3",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=ACCESS_SECRET_KEY,
        region_name="us-east-2"
    )
    bucket = s3_resource.Bucket(name='made-classifier-food-type')

    bucket.download_file(s3_path, path + filepath)



def get_logger():
    logger=logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
    
    stream_logging = logging.StreamHandler(sys.stdout)
    stream_logging.setFormatter(formatter)
    stream_logging.setLevel(logging.INFO)

    file_logging = RotatingFileHandler('logs/app.log', 'w', maxBytes=1024*5, backupCount=2, encoding='utf-8')
    file_logging.setFormatter(formatter)
    file_logging.setLevel(logging.INFO)

    file_logging_error = RotatingFileHandler('logs/app_error.log', 'w', maxBytes=1024*5, backupCount=2, encoding='utf-8')
    file_logging_error.setFormatter(formatter)
    file_logging_error.setLevel(logging.ERROR)

    logger.addHandler(stream_logging)
    logger.addHandler(file_logging)
    logger.addHandler(file_logging_error)

    return logger    

