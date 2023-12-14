def build_positional_index(documents):
    document_num = 1
    pos_index = {}

    for document in documents:
        for positional, term in enumerate(document):
            if term in pos_index:
                pos_index[term][0] += 1
                if document_num in pos_index[term][1]:
                    pos_index[term][1][document_num].append(positional)
                else:
                    pos_index[term][1][document_num] = [positional]
            else:
                pos_index[term] = [1, {document_num: [positional]}]
        document_num += 1

    return pos_index