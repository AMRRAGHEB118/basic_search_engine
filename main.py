from positional_index import create_positional_index, display_positional_index
from tokenization_stemming import apply_tokenization_and_stemming


def main():
    terms = []
    positional_index = {}
    terms = apply_tokenization_and_stemming(terms)
    positional_index = create_positional_index(terms)
    display_positional_index(positional_index)


if __name__ == "__main__":
    main()
