import numpy as np
import pandas as pd

import re
import spacy
import string
from textblob import TextBlob
from textstat import textstat
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import vstack
from sklearn.decomposition import TruncatedSVD

nlp = spacy.load('en_core_web_sm')
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
pd.set_option('display.max_columns', None)



def relative_length(df: pd.DataFrame, prompt_id: str, length: int) -> float:
    rel_length = length / len(df[df.prompt_id == prompt_id]['prompt'].values[0])
    return rel_length


# Count the stop words in the text
def count_stopwords(text: str) -> int:
    stopword_list = set(stopwords.words('english'))
    words = text.split()
    stopwords_count = sum(1 for word in words if word.lower() in stopword_list)
    return stopwords_count

# Count the punctuations in the text
# punctuation_set -> !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
def count_punctuation(text: str) -> int:
    punctuation_set = set(string.punctuation)
    punctuation_count = sum(1 for char in text if char in punctuation_set)
    return punctuation_count

# Count the digits in the text
def count_numbers(text: str) -> int:
    numbers = re.findall(r'\d+', text)
    numbers_count = len(numbers)
    return numbers_count

dict_tags = {'verb': ['VB', 'VBZ', 'VBP', 'VBD', 'VBN', 'VBG'],
             'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
             'adj': ['JJ', 'JJR', 'JJS']}

# Clean and lemmatize the text
def lemmatize_text(text: str) -> list:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = []
    for word in filtered_text:
        pos_tag = nltk.pos_tag([word])[0][1]
        if len(word) > 2:
            if pos_tag in dict_tags['verb']:
                lemmatized_text.append(lemmatizer.lemmatize(word, pos='v'))
            elif pos_tag in dict_tags['noun']:
                lemmatized_text.append(lemmatizer.lemmatize(word, pos='n'))
            elif pos_tag in dict_tags['adj']:
                lemmatized_text.append(lemmatizer.lemmatize(word, pos='a'))
            else:
                lemmatized_text.append(lemmatizer.lemmatize(word))
    return lemmatized_text

# This function applies lemmatize_text on text features
def lemmatize(dataframe : pd.DataFrame, features: str | list) -> pd.DataFrame:
    if isinstance(features, str):
        features = [features]
    for feature in features:
        dataframe[f'{feature}_lemmatized'] = dataframe[feature].apply(lemmatize_text)
    return dataframe

# Count unique words in the lemmatized text
def count_unique_words(words: list) -> int:
    if not isinstance(words, list):
        raise TypeError(f"Expected a list, got {type(words)}")
    unique_words_count = len(set(words))
    return unique_words_count


# Trains a vectorizer on a text
def vectorizer(df: pd.DataFrame, vectorizer=CountVectorizer(), n: int = 50) -> dict:
    vectorizer_dict = {}
    for _, row in df.iterrows():
        vectorizer_dict[row['prompt_id']] = {}
        for text in ['prompt', 'prompt_question']:
            lemmatized_text = ' '.join(row[f'{text}_lemmatized'])
            vectorizer.fit([lemmatized_text])
            word_counts = pd.DataFrame(vectorizer.transform([lemmatized_text]).toarray(), columns=vectorizer.get_feature_names_out())
            top_n_words = word_counts.sum().nlargest(n).index
            top_n_vectorizer = CountVectorizer(vocabulary=top_n_words)
            vectorizer_dict[row['prompt_id']][text] = top_n_vectorizer.fit([lemmatized_text])
    return vectorizer_dict


# Transform a list in a vector
def vectorize(lst: list, vectorizer: CountVectorizer) -> np.ndarray:
    if not isinstance(lst, list):
        raise TypeError(f"Expected a list, got {type(lst)}")
    sparse_matrix = vectorizer.transform([' '.join(lst)])
    vector = sparse_matrix.toarray()
    return vector

# Calculates Jaccard Similarity between two vectors
def jaccard_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    binary_vec1 = [1 if i else 0 for i in vec1[0]]
    binary_vec2 = [1 if i else 0 for i in vec2[0]]
    return jaccard_score(binary_vec1, binary_vec2)


# Extract NER
def ner(text: str) -> pd.DataFrame:
    vector = [(ent.text, ent.label_) for ent in nlp(text).ents]
    return vector

def jaccard_similarity_ner(list1: list[tuple[str]], list2: list[tuple[str]]) -> float:
    if not (isinstance(list1, list) and isinstance(list2, list)):
        raise TypeError(f"Expected lists, got {type(list1)} and {type(list2)}")
    set1 = set(list1)
    set2 = set(list2)
    union_set = set1.union(set2)
    binary_vec1 = [1 if i in set1 else 0 for i in union_set]
    binary_vec2 = [1 if i in set2 else 0 for i in union_set]
    return jaccard_score(binary_vec1, binary_vec2)


def readability(text: str) -> pd.Series:
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    gunning_fog = textstat.gunning_fog(text)
    ari = textstat.automated_readability_index(text)
    return pd.Series([flesch_reading_ease, gunning_fog, ari])


def cosine_similarity_sentiment(a: tuple[float], b: tuple[float]) -> float:
    if np.linalg.norm(a)*np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def sentiment(df: pd.DataFrame, text: str, prompt_id: str) -> float:
    text_blob = TextBlob(text)
    prompt_blob = df[df.prompt_id == prompt_id]['blob'].values[0]
    similarity = cosine_similarity_sentiment(prompt_blob.sentiment, text_blob.sentiment)
    return similarity

def tokenize(text: str) -> str:
    return ' '.join([tag for word, tag in nltk.pos_tag(nltk.word_tokenize(text))])


def tfidf_vectorizer(df_prompt: pd.DataFrame, df_summaries: pd.DataFrame) -> dict[TfidfVectorizer]:
    texts = df_prompt['tokens'].tolist() + df_summaries['tokens'].tolist()
    vectorizer_tfidf = TfidfVectorizer(ngram_range=(4,4))
    vectorizer_tfidf.fit(texts)
    return vectorizer_tfidf


def tfidf(text: str, vectorizer_tfidf: TfidfVectorizer) -> np.ndarray:
    vectors = vectorizer_tfidf.transform([text])
    return vectors


def tfidf_reductor(df: pd.DataFrame, target: str, n_components: int = 50) -> TruncatedSVD:
    tfidf_matrix = vstack(df['tfidf_vector'].values.tolist())
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(tfidf_matrix, df[target])
    return svd


def vector_to_df(df: pd.DataFrame, reductor: TruncatedSVD) -> pd.DataFrame:
    reduced_vectors = reductor.transform(vstack(df['tfidf_vector'].values.tolist()))
    return pd.DataFrame(reduced_vectors)


def tfidf_to_features(df: pd.DataFrame, df_train: pd.DataFrame, target: str, threshold: float = 0.05, n_components=500) -> pd.DataFrame:
    reductor = tfidf_reductor(df, target, n_components=n_components)
    df_reduced = pd.concat([df[[target]], vector_to_df(df_train, reductor)], axis=1).rename(columns={i : str(i) for i in range(n_components)})
    corr = df_reduced[[target] + [str(i) for i in range(n_components)]].corr()
    corr = corr[target].drop(target).to_frame()
    corr = corr[corr[target].abs() >= threshold]
    selected_components = corr.index.tolist()
    df_filtered = pd.concat([df, df_reduced[selected_components]], axis=1)
    return df_filtered, selected_components
