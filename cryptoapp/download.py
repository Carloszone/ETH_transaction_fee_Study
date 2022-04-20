import boto3
import pandas as pd
import os
from zipfile import *

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from werkzeug.security import check_password_hash, generate_password_hash

from pythonProject.db import get_db

bp = Blueprint('download', __name__)

s3_r = boto3.resource("s3")
s3_c = boto3.client('s3')

bucket_name = 'carlos-cryptocurrency-research-project'

bucket_path = 's3://carlos-cryptocurrency-research-project/'
price_path = 'data/price_data/'
transaction_path = 'data/transaction_data/'

def list_file(data_type):
    if data_type == 'price':
        response = s3_c.list_objects_v2(
            Bucket=bucket_name,
            Prefix=price_path)
    elif data_type == 'transaction':
        response = s3_c.list_objects_v2(
            Bucket=bucket_name,
            Prefix=transaction_path)
    else:
        return print(f'error: cannot find the file type: {data_type}')

    files = []
    for content in response.get('Contents', []):
        files.append(content["Key"])
    return files

def read_file(file_name, bucket_path = bucket_path):
    file_path = bucket_path + file_name
    try:
        df = pd.read_csv(file_path)
        return df
    except:
        print('error: cannot find the file')

def zip_file(files_list):
    with zipfile('price_data', 'w') as zipped_file:
        for file in files:
            price_zip.write(s3_c.get_object(Bucket = bucket_name, Key = file))
    return zipped_file

@app.route('/download', methods=['GET', 'POST'])
def download_file(data_type, bucket_name = bucket_name):
    bucket = s3_r.Bucket(bucket_name)
    if data_type == 'price':
        for obj in bucket.objects.filter(Prefix = price_path):
            if not os.path.exists(os.path.dirname(obj.key)):
                os.makedirs(os.path.dirname(obj.key))
            bucket.download_file(obj.key, obj.key)
    elif data_type == 'transaction':
        for obj in bucket.objects.filter(Prefix = transaction_path):
            if not os.path.exists(os.path.dirname(obj.key)):
                os.makedirs(os.path.dirname(obj.key))
            bucket.download_file(obj.key, obj.key)

def upload_file(filename, bucket_path, bucket_name = bucket_name):
    saved_path = bucket_path + filename
    s3_c.upload_file(filename, bucket_name, saved_path)

print('start run')
download_file(data_type = 'price', bucket_name = bucket_name)
print('end')