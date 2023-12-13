from math import log10
import numpy as np
import pandas as pd

def get_tf(document, all_terms):
    wordDict = dict.fromkeys(all_terms, 0)
    for word in document.split():
        wordDict[word] += 1
    return wordDict

def compute_tf(documents):
    all_terms = list(set(term for doc in documents for term in doc.split()))
    tf = pd.DataFrame(get_tf(documents[0], all_terms).values(), index=all_terms)
    for i in range(1, len(documents)):
        tf[i] = get_tf(documents[i], all_terms).values()
    tf.columns = ['doc'+str(i) for i in range(1, 11)]
    return tf

def weighted_tf(x):
    if x > 0:
        return log10(x) + 1
    return 0

def apply_weighted_tf(documents, tf):
    w_tf = tf.copy()
    for i in range(0, len(documents)):
        w_tf['doc'+str(i+1)] = tf['doc'+str(i+1)].apply(weighted_tf)
    return w_tf

def compute_df_idf(w_tf, tf):
    tdf = pd.DataFrame(columns=['df', 'idf'])
    for i in range(len(tf)):
        in_term = w_tf.iloc[i].values.sum()
        tdf.loc[i, 'df'] = in_term
        tdf.loc[i, 'idf'] = log10(10 / (float(in_term)))
    tdf.index=w_tf.index

    return tdf

def compute_tf_idf(w_tf, tdf):
    tf_idf = w_tf.multiply(tdf['idf'], axis=0)
    return tf_idf


def get_doc_len(col):
    return np.sqrt(col.apply(lambda x: x**2).sum())

def get_docs_len(tf_idf):
    docs_len = {}
    for col in tf_idf.columns:
        docs_len[col] = get_doc_len(tf_idf[col])
    return docs_len

def get_normalized_tf_idf(docs_len, col, x):
    try:
        return x / docs_len[col]
    except:
        return 0

def compute_normalized_tf_idf(tf_idf, docs_len):
    normalized_tf_idf = pd.DataFrame()
    for col in tf_idf.columns:
        normalized_tf_idf[col] = tf_idf[col].apply(lambda x : get_normalized_tf_idf(docs_len, col, x))
    return normalized_tf_idf