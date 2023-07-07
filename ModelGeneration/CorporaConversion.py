# general imports
import json
import os

# gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.models import TfidfModel
# import load_data/write_data from LoadWrite
from LoadWrite import load_data, write_data

# Folder location for formatted data output
output_path = 'DataOutput/'
# Name of input data words
data_words_name = 'MilitaryDataWords.json'
# Name of output corpus and id2word
output_corpus_name = 'MilitaryCorpus.json'
output_id2word_name = 'MilitaryId2Word.idw'
# increasing this value will increase the amount of words that are dropped because they are too frequent (according
# to TF-IDF)
low_value = 0.035



# load the data words
data_words =  load_data(output_path + data_words_name)

# TF-IDF REMOVAL

# automatically gets rid of some of the highest occuring words from the corpus. This is done to get rid of words that aren't stop
# words, but don't nessecarily add much to acually discovering what topics are present. (For example, say the word "say")

id2word = corpora.Dictionary(data_words)
corpus = [id2word.doc2bow(text) for text in data_words]

tfidf = TfidfModel(corpus, id2word=id2word)

words = []
words_missing_in_tfidf = []

for i in range(0, len(corpus)):
    bow = corpus[i]
    low_value_words = []
    tfidf_ids = [id for id, value in tfidf[bow]]
    bow_ids = [id for id, value in bow]
    low_value_words = [id for id, value in tfidf[bow] if value < low_value]
    drops = low_value_words + words_missing_in_tfidf
    for item in drops:
        words.append(id2word[item])
    words_missing_in_tfidf = [id for id in bow_ids if
                              id not in tfidf_ids]  # The words with tf-idf score 0 will be missing

    new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
    corpus[i] = new_bow

# save the corpus as json
write_data(output_path + output_corpus_name, corpus)
id2word.save(output_path + output_id2word_name)
print("Saved corpus as " + output_path + output_corpus_name)
