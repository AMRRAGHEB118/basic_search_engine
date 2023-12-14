from math import sqrt
import pandas as pd
from tf_idf import calculate_tf_weight
import re


def process_phrase_query(phrase_query, pos_index):
    final_set = set()
    for word in phrase_query.split():
        if word in pos_index:
            final_set.update(pos_index[word][1].keys())
    return final_set

# Update the process_boolean_query function
def process_boolean_query(boolean_query, pos_index):
    boolean_query = re.sub(r'\s+', ' ', boolean_query)
    boolean_query = boolean_query.strip().lower()

    if ' and ' in boolean_query:
        terms = boolean_query.split(' and ')
        results = [process_phrase_query(term, pos_index) for term in terms]
        final_result = set.intersection(*map(set, results))
        matching_terms = set(terms)

    elif ' or ' in boolean_query:
        terms = boolean_query.split(' or ')
        results = [process_phrase_query(term, pos_index) for term in terms]
        final_result = set.union(*map(set, results))
        matching_terms = set(terms)

    elif ' not ' in boolean_query:
        terms = boolean_query.split(' not ')
        if len(terms) == 2:
            included_docs = process_phrase_query(terms[0], pos_index)
            excluded_docs = process_phrase_query(terms[1], pos_index)
            final_result = included_docs - excluded_docs
            matching_terms = set(terms)
        else:
            raise ValueError("Invalid NOT operator usage")

    else:
        final_result = process_phrase_query(boolean_query, pos_index)
        matching_terms = set([boolean_query])

    return list(final_result), matching_terms



def calculate_cosine_similarity(normalized_tf_idf, query_string, tfd, pos_index):
    if ' and ' in query_string or ' or ' in query_string or ' not ' in query_string:

        boolean_results, matching_terms = process_boolean_query(query_string, pos_index)
        
        query = pd.DataFrame(index=normalized_tf_idf.index)
        query['tf'] = [1 if x in matching_terms else 0 for x in list(normalized_tf_idf.index)]

        if not boolean_results:
            raise ValueError(f'Error: Boolean query "{query_string}" did not match any documents.')

        query = pd.DataFrame(index=normalized_tf_idf.index)
        query['tf'] = [1 if x in boolean_results else 0 for x in list(normalized_tf_idf.index)]
    else:
        # It's a phrase query
        query = pd.DataFrame(index=normalized_tf_idf.index)
        query['tf'] = [1 if x in query_string.split() else 0 for x in list(normalized_tf_idf.index)]

    query['w_tf'] = query['tf'].apply(calculate_tf_weight)

    if query['w_tf'].sum() == 0:
        raise ValueError('Error: Query does not match any terms in the documents.')

    product = normalized_tf_idf.multiply(query['w_tf'], axis=0)
    query['idf'] = tfd['idf'] * query['w_tf']
    query['tf-idf'] = query['w_tf'] * query['idf']
    query['norm'] = query['idf'] / sqrt(sum(query['idf'].values ** 2))

    return product, query


def calculate_query_length(phrase_query, tfd):
    return sqrt(sum([x ** 2 for x in tfd['idf'].loc[phrase_query.split()]]))

def calculate_final_scores(product, query, phrase_query):
    product2 = product.multiply(query['norm'], axis=0)
    scores = {}
    for col in product2.columns:
        if 0 not in product2[col].loc[phrase_query.split()].values:
            scores[col] = product2[col].sum()

    return scores
