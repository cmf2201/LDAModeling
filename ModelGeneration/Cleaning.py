# general imports
import re
import argparse

# Gensim
import gensim
from gensim.utils import simple_preprocess
from gensim.models import Phrases

# spacy
import spacy

# import load_data/write_data from LoadWrite
from LoadWrite import load_data, write_data

# Parse the arguments given to DataPrep
parser = argparse.ArgumentParser()
parser.add_argument("data_path")
parser.add_argument("output_data_words_path")
parser.add_argument("min_word_length")
args = parser.parse_args()

# load List of articles data
data = load_data(args.data_path)

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
        new = simple_preprocess(text, deacc=True)
        final.append(new)
    return (final)

# function to remove very short words from the articles, as more often than not they do not add much (NO SOURCE I MADE THIS UP)
def remove_short(texts):
    final2 = []
    for article in texts:
        final = []
        for word in article:
            if len(word) >= int(args.min_word_length):
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
write_data(args.output_data_words_path, data_bigrams_trigrams)

# Print folder location that Cleaned data is saved to
print("Data cleaned, saved to " + args.output_data_words_path)