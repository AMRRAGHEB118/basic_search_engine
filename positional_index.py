def create_positional_index(documents):
    positional_index = {}
    for doc_id, term in enumerate(documents, start=1):
        for position, term in enumerate(term, start=1):
            if term not in positional_index:
                positional_index[term] = {
                    'count': 1,
                    'positions': {doc_id: [position]}
                }
            else:
                positional_index[term]['count'] += 1
                if doc_id not in positional_index[term]['positions']:
                    positional_index[term]['positions'][doc_id] = [position]
                else:
                    positional_index[term]['positions'][doc_id].append(position)
    return positional_index



def display_positional_index(positional_index):
    print("Positional Index:" + "\n" + "-" * 20 + "\n")
    for term in sorted(positional_index.keys()):
        info = positional_index[term]
        print(f"<{term}, {info['count']};")
        for doc_id, positions in info['positions'].items():
            print(f"doc{doc_id}: {', '.join(map(str, positions))} ;")
        print(">")
        print("\n")