#import used modules
import numpy as np #note: numpy 1.24.2 is used
import pandas as pd
import matplotlib.pyplot as plt
import json
import glob
import os
import re
import random
import math

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import TfidfModel

# spacy
import spacy
from nltk.corpus import stopwords

import pyLDAvis
import pyLDAvis.gensim

# T-5 Model
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline

# Bertopic
from bertopic import BERTopic


# load data
def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# write data
def write_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# variables
data = []

min_word_length = 3

# Preparing Data
path_to_json_files = '../Data/InternshipData-main/Internship Data ArmyAPI Pull_06222023/'
# get all JSON file names as a list
json_file_names = [filename for filename in os.listdir(path_to_json_files) if filename.endswith('.json')]

# get all data into one big list
for json_file_name in json_file_names:
    with open(os.path.join(path_to_json_files, json_file_name)) as json_file:
        json_text = json.load(json_file)
        data.append(json_text['text'])

print(data[0][0:200])