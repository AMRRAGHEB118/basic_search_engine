from math import sqrt
import pandas as pd
import re

from tf_idf import calculate_tf_weight


def process_phrase_query(phrase_query, pos_index):
    final_set = set()
    for word in phrase_query.split():
        if word in pos_index:
            final_set.update(pos_index[word][1].keys())
    return final_set


def process_boolean_query(boolean_query, pos_index):
    boolean_query = re.sub(r'\s+', ' ', boolean_query)
    boolean_query = boolean_query.strip().lower()

    operators = ['and', 'or', 'not']
    operator_used = None

    for operator in operators:
        if f' {operator} ' in boolean_query:
            operator_used = operator
            break

    if operator_used:
        terms = boolean_query.split(f' {operator_used} ')
        results = [process_phrase_query(term, pos_index) for term in terms]

        if operator_used == 'and':
            final_result = set.intersection(*map(set, results))
        elif operator_used == 'or':
            final_result = set.union(*map(set, results))
        elif operator_used == 'not' and len(terms) == 2:
            final_result = results[0] - results[1]
        else:
            raise ValueError(f"Invalid {operator_used.upper()} operator usage")

        matching_terms = set(terms)

    else:
        final_result = process_phrase_query(boolean_query, pos_index)
        matching_terms = set([boolean_query])

    return list(final_result), matching_terms


def calculate_cosine_similarity(normalized_tf_idf, query_string, tfd, pos_index):
    if any(operator in query_string for operator in [' and ', ' or ', ' not ']):

        boolean_results, matching_terms = process_boolean_query(query_string, pos_index)

        query = pd.DataFrame(index=normalized_tf_idf.index)
        query['tf'] = [1 if x in matching_terms else 0 for x in list(normalized_tf_idf.index)]

        if not boolean_results:
            raise ValueError(f'Error: Boolean query "{query_string}" did not match any documents.')

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