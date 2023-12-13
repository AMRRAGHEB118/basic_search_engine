from phrase_query import insert_query, put_query
from positional_index import create_positional_index, display_positional_index
from tf_idf import apply_weighted_tf, compute_df_idf, compute_normalized_tf_idf, compute_tf, compute_tf_idf, get_docs_len
from tokenization_stemming import apply_tokenization_and_stemming, read_documents
import pandas as pd


def main():
    documents = apply_tokenization_and_stemming()

    positional_index = create_positional_index(documents)
    display_positional_index(positional_index)

    documents = read_documents()

    tf = compute_tf(documents)
    w_tf = apply_weighted_tf(documents, tf)
    df_idf = compute_df_idf(w_tf, tf)
    tf_idf = compute_tf_idf(w_tf, df_idf)
    docs_len = get_docs_len(tf_idf)
    normalized_tf_idf = compute_normalized_tf_idf(tf_idf, docs_len)

    print("Term Frequency:")
    print(pd.DataFrame(tf).fillna(0))
    print("\nWeighted Term Frequency:")
    print(pd.DataFrame(w_tf).fillna(0))
    print("\nDF and IDF:")
    print(df_idf)
    print("\nTF-IDF Matrix:")
    print(tf_idf)
    print("\nDocuments Length:")
    print(docs_len)
    print("\nNormalized Term Frequency:")
    print(normalized_tf_idf)

    while True:
        query = input("Enter a query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        insert_query(query, positional_index, tf_idf, normalized_tf_idf)


if __name__ == "__main__":
    main()
