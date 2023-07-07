# import required files
import json
import os
import argparse
# import load_data/write_data from LoadWrite
from LoadWrite import load_data, write_data

# Parse the arguments given to DataPrep
parser = argparse.ArgumentParser()
parser.add_argument("path_to_json_files")
parser.add_argument("output_data_path")
args = parser.parse_args()

# USER: format the data such that it is in the form: List[articles: str]

# get all JSON file names as a list
json_file_names = [filename for filename in os.listdir(args.path_to_json_files) if filename.endswith('.json')]

data = []
# get all data into one big list
for json_file_name in json_file_names:
    with open(os.path.join(args.path_to_json_files, json_file_name)) as json_file:
        json_text = json.load(json_file)
        data.append(json_text['text'])

# write the data to file
write_data(args.output_data_path, data)