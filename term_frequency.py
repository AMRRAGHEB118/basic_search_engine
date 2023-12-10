def compute_frequency(documents):
    term_frequency = {}
    for doc_id, terms in enumerate(documents, start=1):
        for term in terms:
            if term not in term_frequency:
                term_frequency[term] = {f"Doc{doc_id}": 1}
            else:
                term_frequency[term][f"Doc{doc_id}"] = term_frequency[term].get(f"Doc{doc_id}", 0) + 1
    return term_frequency

def compute_term_frequency(documents):
    term_frequency = {}

    for doc_id, terms in enumerate(documents, start=1):
        total_terms = len(terms)

        for term in terms:
            if term not in term_frequency:
                term_frequency[term] = {f"Doc{doc_id}": round(terms.count(term) / total_terms, 2)}
            else:
                term_frequency[term][f"Doc{doc_id}"] = round(terms.count(term) / total_terms, 2)

    return term_frequency

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


