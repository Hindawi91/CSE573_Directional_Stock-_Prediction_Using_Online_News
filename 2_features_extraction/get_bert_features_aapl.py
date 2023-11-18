import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import pickle

data = 'Aapl'


with open(f'../1_data_pre_processing/{data}LabelledNewsData.pkl', 'rb') as f:
    labelled_amzn_news_df = pickle.load(f)

df =labelled_amzn_news_df

# Load pre-trained BERT model and tokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Assuming 'df' is your DataFrame with a 'tokens' column
text_data = df['tokens'].apply(lambda tokens: ' '.join(tokens))

# Tokenize and obtain embeddings for each document
embeddings = []
for text in text_data:
    # Tokenize the text and convert to tensor
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Forward pass through BERT model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the embeddings for the [CLS] token (the first token in the sequence)
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    # Convert the tensor to a NumPy array and append to the list
    embeddings.append(cls_embedding.numpy())

# Convert the list of embeddings to a NumPy array
embedding_matrix = np.vstack(embeddings)

# Create a DataFrame with the embeddings
df_embeddings = pd.DataFrame(embedding_matrix)

# Concatenate the embeddings with your original DataFrame
df_combined = pd.concat([df, df_embeddings], axis=1)

# Display the DataFrame with embeddings
print(df_combined)

# with open('amzn_bert_features.pkl', 'rb') as f:
#     tokenized_aapl_news_df = pickle.load(f)

df_combined.to_csv(f'{data}_bert_features.csv')

with open(f'{data}_bert_features.pkl', 'wb') as f:
    pickle.dump(df_combined, f)