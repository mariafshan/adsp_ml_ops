import numpy as np

# Upload using the DagsHub client, to a DVC tracked folder also called "data".
# Follow the instructions that appear to authorize the request.
from dagshub import upload_files
ds_local_path = '/Users/miaya/Documents/Documents - Mia’s MacBook Pro/MScA/Fall 23/32021 Machine Learning Operations/Week 2/Assignment 1/athletes.csv'
upload_files(repo = 'mia0397/32021_Assignment1', local_path = ds_local_path, remote_path = 'ds_v1')

from dagshub.data_engine import datasources
ds1 = datasources.create_datasource('mia0397/32021_Assignment1', 'my-datasource', "ds_v1")

# Shortly after datasource creation, you should be able to see detected files.
ds1.head().dataframe

