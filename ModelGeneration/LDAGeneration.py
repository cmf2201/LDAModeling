import pyLDAvis
import pyLDAvis.gensim
import gensim
import gensim.corpora as corpora
import os
import webbrowser
import random
import argparse

# import load_data/write_data from LoadWrite
from LoadWrite import load_data

# Parse the arguments given to DataPrep
parser = argparse.ArgumentParser()
parser.add_argument("corpus_path")
parser.add_argument("id2word_path")
parser.add_argument("output_lda_path")
parser.add_argument("num_of_topics")
parser.add_argument("passes")
args = parser.parse_args()

# load in corpus and id2word
corpus = load_data(args.corpus_path)
id2word = corpora.Dictionary.load(args.id2word_path)

# this little line of code does all of the actual LDA modeling!
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=int(args.num_of_topics),
                                           random_state=random.randint(1, 100),
                                           update_every=1,
                                           chunksize=100,
                                           passes=int(args.passes),
                                           alpha="auto")

lda_model.save(args.output_lda_path)
print("LDA Model created and saved as " + args.output_lda_path)
# pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds="mmds", R=20, sort_topics= False)
pyLDAvis.save_html(vis, 'ModelGeneration/LDAvisualization/LDA_Visualization.html')

# webbrowser.open_new_tab(os.path.abspath('ModelGeneration/LDAvisualization/LDA_Visualization.html'))