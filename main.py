from positional_index import create_positional_index, display_positional_index
from term_frequency import compute_term_frequency, display_term_frequency
from tokenization_stemming import apply_tokenization_and_stemming


def main():
    terms = []
    positional_index = {}
    terms = apply_tokenization_and_stemming(terms)
    positional_index = create_positional_index(terms)
    display_positional_index(positional_index)
    term_frequency = compute_term_frequency(terms)
    display_term_frequency(term_frequency)


if __name__ == "__main__":
    main()
