import xml.etree.ElementTree as ElT
import csv


# Smoking challenge
def extract_smoking_status(file):
    tree = ElT.parse(file)
    root = tree.getroot()
    smoking_label = {}
    n_id, label = None, None
    for elem in root.iter():
        # If RECORD element is present
        # use it as note_id, else set to None
        if elem.tag == 'RECORD':
            n_id = elem.attrib['ID']
        elif elem.tag == "SMOKING":
            label = elem.attrib['STATUS']
        smoking_label[str(n_id)] = label
    return smoking_label


def create_challenge_note(sentences, key_dict, labels, challenge):
    new_to_old = {}
    with open(key_dict, 'r') as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            if r[-1] == challenge:
                new_to_old[str(r[0])] = str(r[1])
    notes = {}
    with open(sentences, 'r') as f:
        rd = csv.reader(f)
        for r in rd:
            if len(r) > 0 and r[1] == challenge:
                notes.setdefault(str(r[0]), list()).append(r[-1])

    chl_dt = []

    for key, val in notes.items():
        chl_dt.append([key, labels[new_to_old[str(key)]], '\n'.join(val)])

    return chl_dt


if __name__ == '__main__':
    # Smoking challenge
    train_labels = extract_smoking_status('./datasets/2006_smoking_status/smokers_surrogate_train_all_version2.xml')
    test_labels = extract_smoking_status('./datasets/2006_smoking_status/smokers_surrogate_test_all_groundtruth_version2.xml')

    chl_train = create_challenge_note('./datasets/n2c2_datasets/train_sentences.txt',
                                      './datasets/n2c2_datasets/train_newk_to_oldk.csv',
                                      train_labels,
                                      challenge='smoking')
    chl_test = create_challenge_note('./datasets/n2c2_datasets/test_sentences.txt',
                                     './datasets/n2c2_datasets/test_newk_to_oldk.csv',
                                     test_labels,
                                     challenge='smoking')
    with open('datasets/2006_smoking_status/train_sentences.txt', 'w') as f:
        wr = csv.writer(f)
        for line in chl_train:
            wr.writerow(line)
    with open('datasets/2006_smoking_status/test_sentences.txt', 'w') as f:
        wr = csv.writer(f)
        for line in chl_test:
            wr.writerow(line)


