import pandas as pd
from math import log10, sqrt
from tokenization_stemming import preprocessing


def insert_query(q, positional_index, tdf, normalized_tf_idf):
    docs_found = put_query(q, positional_index, 2)
    if docs_found == []:
        return "Not Fount"
    new_q = preprocessing(q)
    query = pd.DataFrame(index=normalized_tf_idf.index)
    query['tf'] = [1 if x in new_q else 0 for x in list(normalized_tf_idf.index)]
    query['w_tf'] = query['tf'].apply(lambda x : get_w_tf(x))
    product = normalized_tf_idf.multiply(query['w_tf'], axis=0)
    query['idf'] = tdf['idf'] * query['w_tf']
    query['tf_idf'] = query['w_tf'] * query['idf']
    query['normalized'] = 0
    for i in range(len(query)):
        query['normalized'].iloc[i] = float(query['idf'].iloc[i]) / sqrt(sum(query['idf'].values**2))
    print('Query Details')
    print(query.loc[new_q])
    product2 = product.multiply(query['normalized'], axis=0)
    scores = {}
    for col in put_query(q, positional_index, 2):
            scores[col] = product2[col].sum()
    product_result = product2[list(scores.keys())].loc[new_q]
    print()
    print('Product (query*matched doc)')
    print(product_result)
    print()
    print('product sum')
    print(product_result.sum())
    print()
    print('Query Length')
    q_len = sqrt(sum([x**2 for x in query['idf'].loc[new_q]]))
    print(q_len)
    print()
    print('Cosine Simliarity')
    print(product_result.sum())
    print()
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print('Returned docs')
    for tuple in sorted_scores:
        print(tuple[0], end=" ")


def put_query(q, positional_index, display=1):
    lis = [[] for i in range(10)]
    q = preprocessing(q)
    for term in q:

        if term in positional_index.keys():
            for key in positional_index[term][1].keys():
            
                if lis[key-1] != []:
                    
                    if lis[key-1][-1] == positional_index[term][1][key][0]-1:
                        lis[key-1].append(positional_index[term][1][key][0])
                else:
                    lis[key-1].append(positional_index[term][1][key][0])
    positions = []
    if display==1:
        for pos, list in enumerate(lis, start=1):
            if len(list) == len(q):
                positions.append('document '+str(pos))
        return positions
    else:
        for pos, list in enumerate(lis, start=1):
            if len(list) == len(q):
                positions.append('doc'+str(pos))
        return positions

def get_w_tf(x):
    try:
        return log10(x)+1
    except:
        return 0