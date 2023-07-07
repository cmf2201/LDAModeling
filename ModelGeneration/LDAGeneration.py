import pyLDAvis
import pyLDAvis.gensim
import gensim
import gensim.corpora as corpora
import json
import os
import webbrowser


# Parameters for LDA Model
num_of_topics = 7
# Folder location for LDA Model output
output_path = 'Model/'
# Name of LDA output
lda_name = 'LdaModel'
# Folder location for other Data
input_path = 'DataOutput/'
# Name of input corpus and id2word
corpus_name = 'MilitaryCorpus.json'
id2word_name = 'MilitaryId2Word.idw'

# load data
def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# write data
def write_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# load in corpus and id2word
corpus = load_data(input_path + corpus_name)
id2word = corpora.Dictionary.load(input_path + id2word_name)

# this little line of code does all of the actual LDA modeling!
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_of_topics,
                                           random_state=49,
                                           update_every=1,
                                           chunksize=100,
                                           passes=3,
                                           alpha="auto")

lda_model.save(output_path + lda_name)
print("LDA Model created and saved as " + output_path + lda_name)
# pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds="mmds", R=20, sort_topics= False)
pyLDAvis.save_html(vis, 'LDAvisualization/LDA_Visualization.html')
webbrowser.open_new_tab(os.path.abspath('LDAvisualization/LDA_Visualization.html'))