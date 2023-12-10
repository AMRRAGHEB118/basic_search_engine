import os
import natsort
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

files_name = natsort.natsorted(os.listdir("files"))

def handle_stopwords():
    global stop_words
    stop_words = set(stopwords.words("english"))
    stop_words.remove('in')
    stop_words.remove('to')
    stop_words.remove('where')

def read_file(file):
    with open(file, "r") as f:
        return f.read()

def tokenize(text):
    return word_tokenize(text)

def remove_stop_words(tokens):
    return [word for word in tokens if word.lower() not in stop_words]

def apply_stemmer(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

def process_file(file):
    text = read_file(file)
    tokens = tokenize(text)
    tokens = remove_stop_words(tokens)
    tokens = apply_stemmer(tokens)
    return tokens

def apply_tokenization_and_stemming(terms):
    handle_stopwords()
    for file in files_name:
        tokens = process_file("files/" + file)
        terms.append(tokens)
    print(terms)