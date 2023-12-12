from math import log


def compute_tf(documents):
    term_frequency = {}
    for doc_id, terms in enumerate(documents, start=1):
        for term in terms:
            if term not in term_frequency:
                term_frequency[term] = {f"Doc{doc_id}": 1}
            else:
                term_frequency[term][f"Doc{doc_id}"] = term_frequency[term].get(f"Doc{doc_id}", 0) + 1
    return term_frequency

def compute_normalized_tf(documents):
    term_frequency = {}

    for doc_id, terms in enumerate(documents, start=1):
        total_terms = len(terms)

        for term in terms:
            if term not in term_frequency:
                term_frequency[term] = {f"Doc{doc_id}": round(terms.count(term) / total_terms, 2)}
            else:
                term_frequency[term][f"Doc{doc_id}"] = round(terms.count(term) / total_terms, 2)

    return term_frequency

def compute_idf(documents):
    idf = {}
    total_docs = len(documents)

    for term in set(term for doc in documents for term in doc):
        doc_count = sum(1 for doc in documents if term in doc)
        idf[term] = round(log(total_docs / (1 + doc_count)), 2)

    return idf

def compute_tf_idf(documents, idf):
    tf_idf_matrix = {}

    for doc_id, terms in enumerate(documents, start=1):
        tf_idf_matrix[doc_id] = {}
        total_terms = len(terms)

        for term in set(terms):
            tf = terms.count(term) / total_terms
            tf_idf_matrix[doc_id][term] = round(tf * idf.get(term, 0), 2)

    return tf_idf_matrix
