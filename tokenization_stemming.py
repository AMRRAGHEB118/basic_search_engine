import os
from natsort import natsorted
from nltk.tokenize import word_tokenize


def read_documents(folder_path):
    files = natsorted(os.listdir(folder_path))
    documents = []
    for file in files:
        with open(os.path.join(folder_path, file), 'r') as d:
            document = d.read()
        documents.append(document)
    return documents

def tokenize_and_stem(documents, stop_words, stemmer):
    document_list = []
    for doc in documents:
        tokenized_doc = word_tokenize(doc)
        terms = [stemmer.stem(word) for word in tokenized_doc if word not in stop_words]
        document_list.append(terms)
    return document_list