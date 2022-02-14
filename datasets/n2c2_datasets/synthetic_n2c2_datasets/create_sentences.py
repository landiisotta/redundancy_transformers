#!/usr/bin/env python


def create_s_dictionary(file_input):
    """
    vocab generator
    :param file_input: file name
    :return: list of words to use in replacement
    """
    sentence_d = {}
    last_id = ''
    challenge = ''
    sentences = []
    with file_input as file:
        lines = filter(None, (line.rstrip() for line in file))
        for line in lines:
            line = str(line).rstrip('\n').rsplit(",")
            current_id = line[0]
            if current_id != last_id:
                sentence_d[last_id + ',' + challenge] = sentences
                sentences = [line[2]]
                last_id = current_id
                challenge = line[1]
            else:
                sentences.append(line[2])
        sentence_d[last_id + ',' + challenge] = sentences

    empty_keys = [k for k, v in sentence_d.items() if not v]
    for k in empty_keys:
        del sentence_d[k]

    return sentence_d


#file_name = '/Users/alissavalentine/Charney rotation/project code/input/train_sentences.txt'
#train_file = open(file_name)
#directory = create_s_dictionary(train_file)
# train_file.close()

# lengths = []
# for value in directory.values():
#     for s in value:
#         lengths.append(len(s))
#
# plt.hist(lengths)
