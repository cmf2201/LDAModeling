# general imports
import argparse

# gensim
import gensim.corpora as corpora
from gensim.models import TfidfModel
# import load_data/write_data from LoadWrite
from LoadWrite import load_data, write_data

# Parse the arguments given to DataPrep
parser = argparse.ArgumentParser()
parser.add_argument("data_words_path")
parser.add_argument("output_corpus_path")
parser.add_argument("output_id2word_path")
parser.add_argument("low_value")
args = parser.parse_args()


# load the data words
data_words = load_data(args.data_words_path)

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
    low_value_words = [id for id, value in tfidf[bow] if value < float(args.low_value)]
    drops = low_value_words + words_missing_in_tfidf
    for item in drops:
        words.append(id2word[item])
    words_missing_in_tfidf = [id for id in bow_ids if
                              id not in tfidf_ids]  # The words with tf-idf score 0 will be missing

    new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
    corpus[i] = new_bow

# save the corpus as json
write_data(args.output_corpus_path, corpus)
id2word.save(args.output_id2word_path)
print("Saved corpus as " + args.output_corpus_path)
print("Saved id2words as" + args.output_id2word_path)
