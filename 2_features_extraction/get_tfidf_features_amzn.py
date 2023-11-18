import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

data = 'Amzn'
method = 'tfidf'

with open(f'../1_data_pre_processing/{data}LabelledNewsData.pkl', 'rb') as f:
    labelled_amzn_news_df = pickle.load(f)

df =labelled_amzn_news_df

# Load pre-trained BERT model and tokenizer

df['tokens'] = df['tokens'].apply(eval)
# Convert the list of tokens into space-separated strings
df['tokens_str'] = df['tokens'].apply(lambda x: ' '.join(x))

# Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the 'tokens_str' column
tfidf_matrix = tfidf_vectorizer.fit_transform(df['tokens_str'])

# Convert the TF-IDF matrix to a DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

final_df = pd.concat([df, tfidf_df], axis=1)


# with open('amzn_bert_features.pkl', 'rb') as f:
#     tokenized_aapl_news_df = pickle.load(f)

print(final_df)

final_df.to_csv(f'{data}_{method}_features.csv')



with open(f'{data}_{method}_features.pkl', 'wb') as f:
    pickle.dump(final_df, f)