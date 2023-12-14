from math import log10, log
import numpy as np
import pandas as pd


def calculate_tf(documents):
    all_words = [word for doc in documents for word in doc]
    TF = pd.DataFrame(get_TF(documents[0], all_words).values(), index=get_TF(documents[0], all_words).keys())

    for i in range(1, len(documents)):
        TF[i] = get_TF(documents[i], all_words).values()

    TF.columns = ['doc' + str(i) for i in range(1, 11)]
    return TF

def get_TF(doc, all_words):
    word_found = dict.fromkeys(all_words, 0)
    for word in doc:
        word_found[word] += 1
    return word_found

def calculate_tf_weight(x):
    if x > 0:
        return 1 + log(x)
    return 0

def calculate_idf(TF):
    tfd = pd.DataFrame(columns=['df', 'idf'])
    for i in range(len(TF)):
        frequency = TF.iloc[i].values.sum()
        tfd.loc[i, 'df'] = frequency
        tfd.loc[i, 'idf'] = log10(10 / float(frequency))
    tfd.index = TF.index
    return tfd

def calculate_tf_idf(TF, tfd):
    tf_idf = TF.multiply(tfd['idf'], axis=0)
    return tf_idf

def calculate_doc_length(tf_idf):
    doc_length = pd.DataFrame()
    for column in tf_idf.columns:
        doc_length.loc[0, column + '_len'] = np.sqrt(tf_idf[column].apply(lambda x: x ** 2).sum())
    return doc_length

def normalize_tf_idf(tf_idf, doc_length):
    normalized_tf_idf = pd.DataFrame()
    for column in tf_idf.columns:
        normalized_tf_idf[column] = tf_idf[column].apply(lambda x: x / doc_length[column + '_len'].values[0] if
                                                        doc_length[column + '_len'].values[0] != 0 else 0)
    return normalized_tf_idf