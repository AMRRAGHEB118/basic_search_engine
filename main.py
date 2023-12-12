from display_results import display_idf, display_normalized_term_frequency, display_term_frequency, display_tf_idf_matrix
from phrase_query import handle_phrase_query, run_phrase_query
from positional_index import create_positional_index, display_positional_index
from tf_idf import compute_idf, compute_normalized_tf, compute_tf, compute_tf_idf
from tokenization_stemming import apply_tokenization_and_stemming


def main():
    terms = []
    positional_index = {}
    
    # Tokenization and stemming
    terms = apply_tokenization_and_stemming(terms)

    # Constructing auxiliary structure(s)
    positional_index = create_positional_index(terms)
    display_positional_index(positional_index)

    # Term Frequency (TF)
    term_frequency = compute_tf(terms)
    display_term_frequency(term_frequency)

    # Normalized Term Frequency (TF)
    normalized_term_frequency = compute_normalized_tf(terms)
    display_normalized_term_frequency(normalized_term_frequency)

    # Inverse Document Frequency (IDF)
    idf = compute_idf(terms)
    display_idf(idf)

    # TF-IDF matrix
    tf_idf_matrix = compute_tf_idf(terms, idf)
    display_tf_idf_matrix(tf_idf_matrix)

    # Phrase query
    run_phrase_query(positional_index)


if __name__ == "__main__":
    main()
