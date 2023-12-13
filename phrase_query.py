import pandas as pd
from math import log10, sqrt
from tokenization_stemming import preprocessing


def insert_query(q, positional_index, tdf, normalized_tf_idf):
    docs_found = put_query(q, positional_index, 2)
    if docs_found == []:
        return "Not Found"
    
    new_q = preprocessing(q)
    query = pd.DataFrame(index=normalized_tf_idf.index)
    query['tf'] = [1 if x in new_q else 0 for x in list(normalized_tf_idf.index)]
    query['w_tf'] = query['tf'].apply(lambda x: get_w_tf(x))
    query['idf'] = tdf['idf'] * query['w_tf']
    query['tf_idf'] = query['w_tf'] * query['idf']
    query['normalized'] = 0
    
    for i in range(len(query)):
        query.loc[query.index[i], 'normalized'] = float(query['idf'].iloc[i]) / sqrt(sum(query['idf'].values**2)) if sum(query['idf'].values**2) != 0 else 0
    
    print('Query Details')
    print(query.loc[new_q])
    
    product = normalized_tf_idf.multiply(query['w_tf'], axis=0)
    product_result = product.multiply(query['normalized'], axis=0)
    
    print()
    print('Product (query*matched doc)')
    print(product_result.loc[new_q])
    print()
    print('Product Sum')
    print(product_result.sum())
    print()
    print('Query Length')
    q_len = sqrt(sum([x**2 for x in query['idf'].loc[new_q]]))
    print(q_len)
    print()
    print('Cosine Similarity')
    print(product_result.sum())
    print()
    
    scores = {}
    for col in put_query(q, positional_index, 2):
        scores[col] = product_result[col].sum()
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print('Returned docs')
    for tuple in sorted_scores:
        print(tuple[0], end=" ")


def put_query(q, positional_index, display=1):
    lis = [[] for i in range(10)]
    q = preprocessing(q)

    for term in q:
        term_lower = term.lower()  # Ensure lowercase for case-insensitive matching
        if term_lower in positional_index.keys():
            for doc_id, positions in positional_index[term_lower]['positions'].items():
                if lis[doc_id - 1] != []:
                    if lis[doc_id - 1][-1] == positions[0] - 1:
                        lis[doc_id - 1].append(positions[0])
                else:
                    lis[doc_id - 1].append(positions[0])

    positions = []

    if display == 1:
        for pos, lst in enumerate(lis, start=1):
            if len(lst) == len(q):
                positions.append('document ' + str(pos))
        return positions
    else:
        for pos, lst in enumerate(lis, start=1):
            if len(lst) == len(q):
                positions.append('doc' + str(pos))
        return positions

def get_w_tf(x):
    try:
        return log10(x) + 1
    except:
        return 0