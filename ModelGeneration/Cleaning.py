# general imports
import json
import re

# Gensim
import gensim
from gensim.utils import simple_preprocess
from gensim.models import Phrases

# spacy
import spacy


# load data
def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# write data
def write_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# Folder location for formatted data output
output_path = 'DataOutput/'
# Name of input data file
data_name = 'MilitaryData.json'
# Name of output cleaned list of words
output_data_name = 'MilitaryDataWords.json'
# minimum word length
min_word_length = 3

# load List of articles data
data = load_data(output_path + data_name)

# lemmatization function isolates important words using the spacy library
def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts_out = []
    for text in texts:
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    return (texts_out)

# use the gensim simple_preprocess function to preform a simple preprocess on data
def gen_words(texts):
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return (final)

# function to remove very short words from the articles, as more often than not they do not add much (NO SOURCE I MADE THIS UP)
def remove_short(texts):
    final2 = []
    for article in texts:
        final = []
        for word in article:
            if len(word) >= min_word_length:
               final.append(word)
        final2.append(final)
    return final2

# function that will generalize battalion numbers instead of getting rid of them entriely (experimental)
def generalize_numbers(texts):
    final2 = []
    for article in texts:
        final = []
        for word in article:
            word = re.sub(r'^\d+(\w+)',r'numst',word)
            final.append(word)
        final2.append(final)
    return final2
# clean data
lemmatized_texts = lemmatization(data)
data_words = generalize_numbers([article.split() for article in lemmatized_texts])
data_words = remove_short(data_words)
data_words = gen_words([' '.join(article) for article in data_words])


# Bigrams and Trigrams
bigram_phrases = gensim.models.Phrases(data_words, min_count=3, threshold=5)
trigram_phrases = gensim.models.Phrases(bigram_phrases[data_words], threshold=5)

bigram = gensim.models.phrases.Phraser(bigram_phrases)
trigram = gensim.models.phrases.Phraser(trigram_phrases)

def make_bigrams(texts):
    return ([bigram[doc] for doc in texts])

def make_trigrams(texts):
    return ([trigram[bigram[doc]] for doc in texts])

data_bigrams = make_bigrams(data_words)
data_bigrams_trigrams = make_trigrams(data_bigrams)

# Save the cleaned data to output folder
write_data(output_path + output_data_name, data_bigrams_trigrams)

# TODO: Add so output folder is printed
print("Data cleaned, saved to ")