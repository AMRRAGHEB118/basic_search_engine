from math import log

def display_term_frequency(term_frequency):
    terms = sorted(term_frequency.keys())
    header = ["Term"] + [f"Doc{i}" for i in range(1, 11)]
    separator1 = "-" * (sum(len(col) + 6 for col in header) - 1)
    separator2 = "_" * (sum(len(col) + 6 for col in header) - 1)

    print("{:<15}".format(header[0]), end="")
    for col in header[1:]:
        print("{:<8}".format(col), end="")
    print("\n" + separator1)

    for term in terms:
        frequencies = [str(term_frequency[term].get(f"Doc{i}", 0)) for i in range(1, 11)]
        print("{:<15}".format(term), end="")
        for freq in frequencies:
            print("{:<8}".format(freq), end="")
        print("\n")
    print(separator2)
    print("\n")

def display_normalized_term_frequency(term_frequency):
    terms = sorted(term_frequency.keys())
    header = ["Term"] + [f"Doc{i}" for i in range(1, 11)]
    separator1 = "-" * (sum(len(col) + 6 for col in header) - 1)
    separator2 = "_" * (sum(len(col) + 6 for col in header) - 1)

    print("{:<15}".format(header[0]), end="")
    for col in header[1:]:
        print("{:<8}".format(col), end="")
    print("\n" + separator1)

    for term in terms:
        frequencies = [f"{term_frequency[term].get(f'Doc{i}', 0):.2f}" for i in range(1, 11)]
        print("{:<15}".format(term), end="")
        for freq in frequencies:
            print("{:<8}".format(freq), end="")
        print("\n")
    print(separator2)
    print("\n")

def display_idf(idf):
    terms = sorted(idf.keys())
    header = ["Term", "IDF"]
    separator = "-" * (sum(len(col) + 6 for col in header) - 1)

    print("{:<15}{:<8}".format(header[0], header[1]))
    print(separator)

    for term in terms:
        print("{:<15}{:<8.2f}".format(term, idf[term]))
    print("\n")

def display_tf_idf_matrix(tf_idf_matrix):
    terms = set(term for doc in tf_idf_matrix.values() for term in doc)
    header = ["Term"] + [f"Doc{i}" for i in range(1, 11)]
    separator1 = "-" * (sum(len(col) + 6 for col in header) - 1)

    print("{:<15}".format(header[0]), end="")
    for col in header[1:]:
        print("{:<8}".format(col), end="")
    print("\n" + separator1)

    for term in terms:
        scores = [f"{tf_idf_matrix.get(doc, {}).get(term, 0):.2f}" for doc in range(1, 11)]
        print("{:<15}".format(term), end="")
        for score in scores:
            print("{:<8}".format(score), end="")
        print("\n")

    print("\n")
