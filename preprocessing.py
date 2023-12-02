import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from gensim.models import Word2Vec
import numpy as np
import pandas as pd

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize the tokenizer (choose the appropriate model)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def preprocess(text):
    """
    Preprocess the text for BERT model.
    Args:
        text (str): Original text.
    Returns:
        str: Preprocessed text.
    """

    # Lower casing
    text = text.lower()

    # Remove URL
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove user @ references from tweet (keep hashtags if they add context)
    text = re.sub(r'\@\w+', '', text)

    # Replacing consecutive letters more than 2 times with a single instance
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # No need for further tokenization, stop word removal, and lemmatization for BERT
    return text


# def preprocess(text):
#     # Lower casing
#     text = text.lower()

#     # Remove URL
#     text = re.sub(r'http\S+|www\S+', '', text)

#     # Remove user @ references and '#' from tweet
#     text = re.sub(r'\@\w+|\#', '', text)

#     # Replacing consecutive letters
#     text = re.sub(r'(.)\1{2,}', r'\1', text)

#     # Remove punctuation from each text
#     table = str.maketrans('', '', string.punctuation)
#     text = text.translate(table)

#     # Remove extra spaces
#     text = re.sub(r'\s+', ' ', text).strip()

#     # Tokenize text
#     word_tokens = word_tokenize(text)

#     # Filter out stop words
#     stop_words_set = set(stopwords.words('english'))
#     filtered_tokens = [
#         word for word in word_tokens if word.isalpha() and word not in stop_words_set]

#     lemmatizer = WordNetLemmatizer()
#     stemmer = PorterStemmer()

#     tokens = [stemmer.stem(lemmatizer.lemmatize(token))
#               for token in filtered_tokens]

#     # Reconstruct the text to feed into BERT tokenizer
#     tokens = ' '.join(tokens)

#     return tokens


def convert_labels_to_numeric(df, label_columns):
    """
    Convert binary labels ('n'/'y') to numeric (0/1).

    Args:
    df (pandas.DataFrame): DataFrame containing the label columns.
    label_columns (list): List of columns in the DataFrame that contain the labels.

    Returns:
    pandas.DataFrame: DataFrame with converted numeric labels.
    """
    def map_labels(label):
        return 1 if label == 'y' else 0

    for col in label_columns:
        df[col] = df[col].apply(map_labels)

    return df


def preprocess_text(raw_text):
    # Tokenize the text
    tokens = word_tokenize(raw_text)

    # Lowercase all the tokens
    tokens = [token.lower() for token in tokens]

    # Remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [token.translate(table) for token in tokens]

    # Filter out stop words and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha()
              and token not in stop_words]

    # Lemmatize each token
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    # Lemmatize and then stem each token
    tokens = [stemmer.stem(lemmatizer.lemmatize(token)) for token in tokens]

    return tokens


def word2vec(df):
    # Train Word2Vec model
    token_lists = df['processed_Text'].tolist()
    word2vec_model = Word2Vec(token_lists, vector_size=100,
                              window=5, min_count=1, workers=4)

    # Aggregate Word Embeddings

    def document_vector(word2vec_model, doc):
        doc = [word for word in doc if word in word2vec_model.wv.key_to_index]
        return np.zeros(word2vec_model.vector_size) if len(doc) == 0 else np.mean(word2vec_model.wv[doc], axis=0)

    doc_vectors = np.array([document_vector(word2vec_model, doc)
                            for doc in token_lists])
    word2vec_df = pd.DataFrame(doc_vectors)

    # Merge with labels
    df_labels = df[["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]]
    combined_df = pd.concat([word2vec_df, df_labels], axis=1)

    # Save the combined DataFrame
    combined_df.to_csv("Word2Vec_Dataset.csv",
                       encoding='utf-8', index=False)
