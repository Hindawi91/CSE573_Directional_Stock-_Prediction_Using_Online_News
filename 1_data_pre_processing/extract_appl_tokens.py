import pandas
import os
import pandas as pd
import nltk
import json
from datetime import datetime, timezone
from nltk.corpus import stopwords, words
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer
import string
import re

NEWS_DIRECTORY = '/home/local/ASUAD/falhinda/CSE573/data/News/'

# Download libraries
nltk.download('punkt')
nltk.download('words')
nltk.download('stopwords')

def get_news_articles(path):
    news_text = []
    news_publish_time = []
    news_source = []
    domains_to_select = ['ae', 'au', 'bb', 'biz', 'ca', 'in', 'io', 'net', 'uk', 'com']
    
    
    with os.scandir(path) as news_directories:
        for directory in news_directories:
            with os.scandir(os.path.join(path, directory.name)) as folder:
                for article in folder:
                    with open(os.path.join(path, directory.name, article.name), encoding='latin1') as f:
                        try:
                            news_data = json.load(f)
                        except json.JSONDecodeError as e:
                            print(f"Error loading JSON from {article.name}: {e}")
                            continue
                    if 'site' in news_data['thread'] and news_data['thread']['site']:
                        source = news_data['thread']['site'] 
                        domain = source.split('.')[-1]
                        # Skip news from domain not in the list
                        if domain not in domains_to_select:
                            continue
                        news_source.append(source)
                    else:
                        news_source.append(None)
                    if 'published' in news_data and news_data['published']:
                        news_publish_time.append(news_data['published'])
                    else:
                        news_publish_time.append(None)
                    if 'text' in news_data and news_data['text']:
                        news_text.append(news_data['text'])
                    else:
                        news_text.append(None)

    df = pd.DataFrame({
        'timestamp': news_publish_time,
        'text': news_text,
        'source': news_source,
    })




    return df

news_df = get_news_articles(NEWS_DIRECTORY)

# Convert timezone to UTC and drop the timezone
news_df['timestamp'] = news_df['timestamp'].apply(lambda x: datetime.fromisoformat(x).astimezone(tz=timezone.utc))
news_df['timestamp'] = news_df['timestamp'].dt.tz_localize(None)

news_df['text'] = news_df['text'].apply(lambda x: x.lower())

columns = ['timestamp', 'source', 'sentences']
processed_amzn_news_df = pd.DataFrame(columns=columns)
processed_aapl_news_df = pd.DataFrame(columns=columns)

# Extract and separate sentences containing AAPL and AMZN
for index, row in news_df.iterrows():
    text = row['text']
    aapl_sentences, amzn_sentences = [], []
    for sentence in nltk.sent_tokenize(text):
        if 'amazon' in sentence or 'amzn' in sentence:
            amzn_sentences.append(sentence)
        if 'apple' in sentence or 'aapl' in sentence:
            aapl_sentences.append(sentence)
    if aapl_sentences:
        processed_aapl_news_df.loc[len(processed_aapl_news_df)] = [row['timestamp'], row['source'], aapl_sentences]
    if amzn_sentences:
        processed_amzn_news_df.loc[len(processed_amzn_news_df)] = [row['timestamp'], row['source'], amzn_sentences]
        
print(len(processed_amzn_news_df), len(processed_aapl_news_df))

def extract_words(input_words):
    from nltk.corpus import words
    
    # Remove all non-ascii words
    processed_words = [w for w in input_words if w.isascii()]
    
    # Remove punctuation words
    tr_dict = str.maketrans(dict.fromkeys(string.punctuation))
    processed_words = [w.translate(tr_dict) for w in processed_words if w]
    
    # Remove links
    final_words = []
    for word in processed_words:
        if not re.match('[www]', word):
            final_words.append(word)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    processed_words = [w for w in final_words if w not in stop_words]
    
    # Stem words and return unique words
    stemmer = SnowballStemmer('english')
    seen = set()
    processed_words = [stemmer.stem(word) for word in processed_words if word]
    processed_words = [x for x in processed_words if not (x in seen or seen.add(x))]
    del seen
    
    # Keep only words from English dictionary
    english_words = set([w.lower() for w in words.words()])
    processed_words = [w for w in processed_words if w in english_words]
    
    return processed_words

tokenized_df_columns = ['timestamp', 'source', 'tokens']
tokenized_amzn_news_df = pd.DataFrame(columns=tokenized_df_columns)
tokenized_aapl_news_df = pd.DataFrame(columns=tokenized_df_columns)



print("\n\nProcessing %d records" % len(processed_aapl_news_df))
for index, row in processed_aapl_news_df.iterrows():
    # This break is only for testing purpose
#     if index >= 1000:
#         break
    if index % 500 == 0:
        print("Completed %d rows" % index)
    token_words = []
    for sentence in row['sentences']:
        token_words.extend(nltk.wordpunct_tokenize(sentence))
    token_words = extract_words(token_words)
    if index == 0:
        print(token_words)
    tokenized_aapl_news_df.loc[index] = [
        processed_aapl_news_df.loc[index]['timestamp'], 
        processed_aapl_news_df.loc[index]['source'],
        token_words
    ]
tokenized_aapl_news_df.to_csv('AaplExtractedTokens.csv')