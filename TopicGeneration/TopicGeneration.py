import gensim.models
from bertopic import BERTopic
import re
import random
import json
import argparse

# T-5 Model
from transformers import T5Tokenizer, T5ForConditionalGeneration

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Parse the arguments given to Topic Generation
parser = argparse.ArgumentParser()
parser.add_argument("gen_data_path")
parser.add_argument("gen_id2word_path")
parser.add_argument("gen_lda_path")
parser.add_argument("output_topic_path")
parser.add_argument("num_of_topics")
parser.add_argument("num_of_keywords")
parser.add_argument("num_of_headlines")
parser.add_argument("include_all_gen_names")
args = parser.parse_args()


# load data
def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# write data
def write_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


lda_model = gensim.models.LdaModel.load(args.gen_lda_path)
id2word = gensim.corpora.Dictionary.load(args.gen_id2word_path)
data = load_data(args.gen_data_path)

print("Successfully Loaded Model!")

# store the topic keywords
topics = []
for topicNum in range(int(args.num_of_topics)):
    topic_terms = lda_model.get_topic_terms(topicNum, topn=30)
    topic_term_words = [(id2word[id], percent) for id, percent in topic_terms]
    topics.append(topic_term_words)

# create a BerTopic Model to use later to evaluate Topic names
topic_model = BERTopic()
topic_model.fit_transform(data)
print("Fit articles to Bertopic Transform")

# bring in the T5 model
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto")
print("T5 Model Loaded")

# first, use a random selection of X number of keywords to generate 10 possible topics:
def gen_headlines(keywords, number_of_keywords=20, number_of_headlines=10):
    headlines = []
    for i in range(number_of_headlines):
        # randomly generate list with topics that are more relevent being more likely to apperar
        instance_keywords = random.choices([words for words, percent in keywords],
                                           weights=[percent for words, percent in keywords], k=number_of_keywords)
        #         print(topics[0])
        #         print(test_keywords)

        input_text = "create a topic given the following keywords: '" + ", ".join(instance_keywords) + "'"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

        outputs = model.generate(input_ids)

        # then use those topics to create one general topic
        headline = tokenizer.decode(outputs[0])

        # removes the html (<> padding) before and after result
        headline = re.sub(r'^<.{1,5}>|<.{0,5}$', '', headline).strip()
        headlines.append(headline)
    return headlines


def measure_similarity_of_topic(topic_labels):
    embedding = topic_model.embedding_model.embed(topic_labels)
    similarity_matrix = cosine_similarity(embedding)

    top_label = topic_labels[np.argmax(np.sum(similarity_matrix, axis=1))]

    # scoring topic
    triu_mat = np.triu(similarity_matrix, k=1)
    score = np.mean(triu_mat[np.nonzero(triu_mat)])

    return similarity_matrix, top_label, score

print("Generating topic names...")
# generate headlines
topic_names = []
for topic in topics:
    headlines = gen_headlines(topic, number_of_keywords=int(args.num_of_keywords), number_of_headlines=int(args.num_of_headlines))
    # use the similarity to topic score
    simi, top_lab, score = measure_similarity_of_topic(headlines)

    topic_names.append(top_lab)
    print(top_lab)

write_data(args.output_topic_path, topic_names)
print("Successfully generated all topic names!")



