# import required files
import json
import os

# load data
def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# write data
def write_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# Put here the path to the file containing the json data to be used in the LDA model
path_to_json_files = '../Data/InternshipData-main/Internship Data ArmyAPI Pull_06222023/'
# Folder location for formatted data output
output_path = 'DataOutput/'
# Name of output file
data_name = 'MilitaryData.json'

# format the data such that it is in the form: List[articles: str]

# get all JSON file names as a list
json_file_names = [filename for filename in os.listdir(path_to_json_files) if filename.endswith('.json')]

data = []
# get all data into one big list
for json_file_name in json_file_names:
    with open(os.path.join(path_to_json_files, json_file_name)) as json_file:
        json_text = json.load(json_file)
        data.append(json_text['text'])

# write the data to file
write_data(output_path + data_name, data)
print(data[0][0:200])