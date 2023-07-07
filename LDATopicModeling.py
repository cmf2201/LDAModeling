#import used modules

import json
import os
import argparse
import webbrowser
parser = argparse.ArgumentParser()


# Parameters for DataPrep.py
# NOTE: may have to adjust DataPrep.py code to make it work for your json data.
path_to_json_files = 'Data/InternshipData-main/Internship_Data_ArmyAPI_Pull_06222023/'#path to json data to be used in the LDA model
output_data_path = 'ModelGeneration/DataOutput/MilitaryData.json'  # Path for formatted data output
parser.add_argument("-d", "--dataprep", help="prepares raw Json data by converting it to a list of strings",
                    action="store_true")

# Parameters for Cleaning.py
data_path = 'ModelGeneration/DataOutput/MilitaryData.json' # Path for data formatted from DataPrep (should be same as output_data_path)
output_data_words_path = 'ModelGeneration/DataOutput/MilitaryDataWords.json'  # Path for formatted data output
min_word_length = "3" # minimuim word length
parser.add_argument("-cl", "--cleaning", help="cleans raw data by removing stop words, short words, and generally unhelpfully words",
                    action="store_true")

# Parameters for CorporaConversion.py
data_words_path = 'ModelGeneration/DataOutput/MilitaryDataWords.json' # Path for data formatted from Cleaning (should be same as output_data_words_path)
output_corpus_path = 'ModelGeneration/DataOutput/MilitaryCorpus.json' # Path for corpus to be saved in
output_id2word_path = 'ModelGeneration/DataOutput/MilitaryId2Word.id2word' # Path for corpus to be saved in
low_value = "0.035" # The threshold for which to cutoff frequent words in TF-IDF (Higher number, more words cutoff)
parser.add_argument("-co", "--corpora", help="converts the cleaned data into corpus (bank of words) format, and uses TF-IDF to remove frequent words",
                    action="store_true")

# Parameters for LDAGeneration.py
corpus_path = 'ModelGeneration/DataOutput/MilitaryCorpus.json' # corpus path (should be same as output_corpus_path)
id2word_path = 'ModelGeneration/DataOutput/MilitaryId2Word.id2word' # id2word path (should be same as output_id2word_path)
output_lda_path = 'ModelGeneration/Model/MilitaryLDAmodel' # output path for LDA model
num_of_topics = "7" # The number of Topic Bubbles to generate: This is also used in Topic Generation
passes = "3" # The number of passes to do
parser.add_argument("-m", "--modelgen", help="Generates the LDA model",
                    action="store_true")

# Parameter to show webbrowser
parser.add_argument("-di", "--display", help="Displays LDA model HTML",
                    action="store_true")



args = parser.parse_args()

if args.dataprep:
    query1 = "python ModelGeneration/DataPrep.py " + path_to_json_files + " " + output_data_path
    print(query1)
    os.system(query1)

if args.cleaning:
    query2 = "python ModelGeneration/Cleaning.py " + data_path + " " + output_data_words_path + " " + min_word_length
    print(query2)
    os.system(query2)

if args.corpora:
    query3 = "python ModelGeneration/CorporaConversion.py " + data_words_path + " " + output_corpus_path + " " + output_id2word_path + " " + low_value
    print(query3)
    os.system(query3)

if args.modelgen:
    query4 = "python ModelGeneration/LDAGeneration.py " + corpus_path + " " + id2word_path + " " + output_lda_path + " " + num_of_topics + " " + passes
    print(query4)
    os.system(query4)

if args.display:
    webbrowser.open_new_tab(os.path.abspath('ModelGeneration/LDAvisualization/LDA_Visualization.html'))