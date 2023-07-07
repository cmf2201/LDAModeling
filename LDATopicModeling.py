#import used modules

import json
import os
import argparse
parser = argparse.ArgumentParser()


# Parameters for DataPrep.py
# NOTE: may have to adjust DataPrep.py code to make it work for your json data.
path_to_json_files = 'Data/InternshipData-main/Internship_Data_ArmyAPI_Pull_06222023/'#path to json data to be used in the LDA model
output_data_path = 'ModelGeneration/DataOutput/MilitaryData.json'  # Path for formatted data output
parser.add_argument("-d", "--dataprep", help="prepares raw Json data by converting it to a list of strings",
                    action="store_true")


args = parser.parse_args()

if args.dataprep:
    query1 = "python ModelGeneration/DataPrep.py " + path_to_json_files + " " + output_data_path
    print(query1)
    os.system(query1)

