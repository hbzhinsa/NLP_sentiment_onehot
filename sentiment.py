# Text classification using Keras
#%%
import pandas as pd
import nltk
#nltk.download('wordnet')  
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


# %%
# * Try lemmatizing using NLTK to improve prediction
sentence = "The striped bats are hanging on their feet for best"
word_list = nltk.word_tokenize(sentence)
print(word_list)
wnl = WordNetLemmatizer()
lemmatized_output = ' '.join([wnl.lemmatize(w) for w in word_list])
print(lemmatized_output)
print('not so good- -!, Try again')
# %%
# Try better lemmatizing 
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

word = 'are'
print(wnl.lemmatize(word, get_wordnet_pos(word)))
print([wnl.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)])
print('better')

# * sentiment prediction>>
# Training data path
# download data from>
# https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences
#%%
filepath={'yelp':'sentiment labelled sentences/amazon_cells_labelled.txt',
          'amazon': 'sentiment labelled sentences/imdb_labelled.txt',
          'yelp': 'sentiment labelled sentences/yelp_labelled.txt'}
# * read data
df_list = []
for source, filepath in filepath.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source 
    df_list.append(df)
df=pd.concat(df_list)
df.head()

# % Prepare data 
# 
#! https://realpython.com/python-keras-text-classification/
for index in len()
