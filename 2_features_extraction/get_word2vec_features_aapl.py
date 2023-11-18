import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

data = 'Aapl'
method = 'word2vec'

with open(f'../1_data_pre_processing/{data}LabelledNewsData.pkl', 'rb') as f:
    labelled_amzn_news_df = pickle.load(f)

df =labelled_amzn_news_df

# Load pre-trained BERT model and tokenizer

word2vec_model = Word2Vec(sentences=df['tokens'], vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.train(df['tokens'], total_examples=len(df['tokens']), epochs=10)

def average_word_vectors(tokens, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0.

    for word in tokens:
        if word in vocabulary:
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model.wv[word])

    if nwords:
        feature_vector = np.divide(feature_vector, nwords)

    return feature_vector


word_vectors = [average_word_vectors(tokens, word2vec_model, word2vec_model.wv.index_to_key, 100) for tokens in df['tokens']]
word_vectors = np.vstack(word_vectors)

column_names = ['f{}'.format(i) for i in range(1, 101)]
features_df  = pd.DataFrame(word_vectors, columns=column_names)
df_with_word2vec = pd.concat([df, features_df], axis=1)




# with open('amzn_bert_features.pkl', 'rb') as f:
#     tokenized_aapl_news_df = pickle.load(f)

print(df_with_word2vec)

df_with_word2vec.to_csv(f'{data}_{method}_features.csv')



with open(f'{data}_{method}_features.pkl', 'wb') as f:
    pickle.dump(df_with_word2vec, f)