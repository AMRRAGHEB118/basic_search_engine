from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
from phrase_query import calculate_cosine_similarity, calculate_final_scores, calculate_query_length, process_boolean_query
from positional_index import build_positional_index
from tf_idf import calculate_doc_length, calculate_idf, calculate_tf, calculate_tf_idf, normalize_tf_idf
from tokenization_stemming import read_documents, tokenize_and_stem

warnings.filterwarnings("ignore")



def main():
    try:
        print("_______________________________ First Part: Tokenization and Stemming _______________________________")
        english_stops = set(stopwords.words('english'))
        english_stops -= {'in', 'to', 'where'}

        porter = PorterStemmer()

        documents = read_documents('files')
        stemmed_documents = tokenize_and_stem(documents, english_stops, porter)
        print("\nStemmed Documents:")
        for idx, doc in enumerate(stemmed_documents, start=1):
            print(f"Document {idx}: {doc}")

        print("\nTokenization and Stopword Removal:")
        for idx, doc in enumerate(documents, start=1):
            tokenized_doc = word_tokenize(doc)
            non_stemmed_doc = [word for word in tokenized_doc if word not in english_stops]
            print(f"Document {idx} (Original): {doc}")
            print(f"Document {idx} (Tokenized and Stopword Removed): {non_stemmed_doc}\n")

        print("\n_______________________________ Second Part: Positional Index _______________________________")
        pos_index = build_positional_index(stemmed_documents)
        print("\nPositional Index:")
        for term, details in pos_index.items():
            print(f"Term: {term}, Document Frequency: {details[0]}, Positions: {details[1]}\n")

        print("\n_______________________________ Third Part: Phrase Query with Boolean Operators _______________________________")
        boolean_query = input("Enter your boolean query: ")
        boolean_results, matching_terms = process_boolean_query(boolean_query, pos_index)
        print(f"\nDocuments matching the Boolean Query '{boolean_query}':")
        print(boolean_results)

        print("\n_______________________________ Fourth Part: TF-IDF Calculation _______________________________")
        TF = calculate_tf(stemmed_documents)
        print("\nTerm Frequency (TF):")
        print(TF)

        tfd = calculate_idf(TF)
        print("\nInverse Document Frequency (IDF):")
        print(tfd)

        tf_idf = calculate_tf_idf(TF, tfd)
        print("\nTF-IDF:")
        print(tf_idf)

        doc_length = calculate_doc_length(tf_idf)
        print("\nDocument Length:")
        print(doc_length)

        normalized_tf_idf = normalize_tf_idf(tf_idf, doc_length)
        print("\nNormalized TF-IDF:")
        print(normalized_tf_idf)

        print("\n_______________________________ Fifth Part: Cosine Similarity _______________________________")
        try:
            product, query = calculate_cosine_similarity(normalized_tf_idf, boolean_query, tfd, pos_index)
            print("\nQuery Information:")
            print(query.loc[boolean_query.split()])

            query_length = calculate_query_length(boolean_query, tfd)
            print("\nQuery Length:", query_length)

            scores = calculate_final_scores(product, query, boolean_query)
            print("\nCosine Similarity Scores:")
            for doc, score in scores.items():
                print(f"Document {doc}: {score}")

            final_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            print("\nRanked Documents:")
            for doc, score in final_score:
                print(f"Document {doc}: {score}")
                print()

        except ValueError as e:
            print(f'\nError: {e}')

    except Exception as e:
        print(f'\nError: {e}')

if __name__ == '__main__':
    main()
