def handle_phrase_query(positional_index, query):
    terms = query.split()
    matched_docs = set()

    for term in terms:
        if term in positional_index:
            matched_docs.update(positional_index[term]['positions'].keys())
    return matched_docs

def display_matched_docs(matched_docs):
    print("Matched Documents:" + "\n" + "-" * 20 + "\n")
    if matched_docs:
        for doc_id in sorted(matched_docs):
            print(f"doc{doc_id}")
    else:
        print("No documents found for the phrase query.")
    print("\n")

def run_phrase_query(positional_index):
    while True:
        try:
            query = input("Enter a phrase query (or press Ctrl + Z to quit or type 'exit'): ")
            if query.lower() == 'exit':
                break

            matched_docs = handle_phrase_query(positional_index, query)
            display_matched_docs(matched_docs)
        except EOFError:
            print("\nExiting...")
            break
