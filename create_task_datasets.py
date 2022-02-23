import xml.etree.ElementTree as ElT
from collections import OrderedDict
import csv
import sys
import re
import os


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


# 2018 cohort selection challenge
def extract_cohort_status(file_path):
    keys = ["ABDOMINAL", "ADVANCED-CAD", "ALCOHOL-ABUSE",
            "ASP-FOR-MI", "CREATININE", "DIETSUPP-2MOS",
            "DRUG-ABUSE", "ENGLISH", "HBA1C",
            "KETO-1YR", "MAJOR-DIABETES", "MAKES-DECISIONS",
            "MI-6MOS"]
    data = {}
    for file in os.listdir(file_path):
        if not re.match(r'^\.[A-Za-z]', file):
            tags = OrderedDict()
            id_note = file.split('.')[0]
            tree = ElT.parse(os.path.join(file_path, file))
            root = tree.getroot()
            for el in root.iter():
                if el.tag == 'TEXT':
                    n_notes = _count_sep(el.text)
                elif el.tag in keys:
                    tags[el.tag] = el.attrib['met']
            ids = [id_note + '::' + f'0{str(i + 1)}' for i in range(n_notes)]
            data[id_note] = (ids, tags)
    return data


def _count_sep(text):
    c = re.findall(r'\*{100}', text)
    return len(c)


def create_challenge_note(sentences, key_dict, labels, challenge):
    new_to_old, old_to_new = _read_dict_ids(key_dict, challenge)
    notes = _read_sentences(sentences, challenge)

    chl_dt = []
    if challenge == 'smoking':
        for key, val in notes.items():
            chl_dt.append([key, labels[new_to_old[str(key)]], '\n'.join(val)])
    # Implementation for cohort_selection_challenge
    elif challenge == 'long':
        not_found_keys = _concat_notes(chl_dt, labels, notes, old_to_new)
        while len(not_found_keys) >= 1:
            current_file = sentences.split('/')[-1].split('_')
            current_dict = key_dict.split('/')[-1].split('_', 1)
            path = '/'.join(sentences.split('/')[:-1])
            if current_file[0] == 'train':
                new_file = '_'.join(['test', current_file[1]])
                new_dict = '_'.join(['test', current_dict[1]])
            else:
                new_file = '_'.join(['train', current_file[1]])
                new_dict = '_'.join(['train', current_dict[1]])
            new_to_old, old_to_new = _read_dict_ids(os.path.join(path, new_dict), challenge)
            notes = _read_sentences(os.path.join(path, new_file), challenge)
            not_found_keys = _concat_notes(chl_dt, not_found_keys, notes, old_to_new)

    return chl_dt


def _read_dict_ids(key_dict, challenge):
    new_to_old = {}
    old_to_new = {}
    with open(key_dict, 'r') as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            if r[-1] == challenge:
                new_to_old[str(r[0])] = str(r[1])
                old_to_new[str(r[1])] = str(r[0])
    return new_to_old, old_to_new


def _read_sentences(sentences, challenge):
    notes = {}
    with open(sentences, 'r') as f:
        rd = csv.reader(f)
        for r in rd:
            if len(r) > 0 and r[1] == challenge:
                notes.setdefault(str(r[0]), list()).append(r[-1])
    return notes


def _concat_notes(chl_dt, labels, notes, old_to_new):
    not_found_keys = {}
    chk_keys = [el[0] for el in chl_dt]
    for key in labels:
        concat_notes = []
        for long_id in labels[key][0]:
            if long_id in old_to_new or long_id in chk_keys:
                concat_notes.extend(notes[old_to_new[long_id]])
            else:
                if key in not_found_keys:
                    not_found_keys[key][0].append(long_id)
                else:
                    not_found_keys[key] = ([long_id], labels[key][1])
        if len(concat_notes) > 1:
            chl_dt.append([key] +
                          ['::'.join([tag, lab]) for tag, lab in labels[key][1].items()] +
                          ['\n'.join(concat_notes)])
    return not_found_keys


if __name__ == '__main__':
    if sys.argv[1] == 'smoking_challenge':
        # Smoking challenge (2006) -- 5 classes
        train_labels = extract_smoking_status(
            './datasets/2006_smoking_status/smokers_surrogate_train_all_version2.xml')
        test_labels = extract_smoking_status(
            './datasets/2006_smoking_status/smokers_surrogate_test_all_groundtruth_version2.xml')

        chl_train = create_challenge_note('./datasets/n2c2_datasets/train_sentences.txt',
                                          './datasets/n2c2_datasets/train_newk_to_oldk.csv',
                                          train_labels,
                                          challenge='smoking')
        chl_test = create_challenge_note('./datasets/n2c2_datasets/test_sentences.txt',
                                         './datasets/n2c2_datasets/test_newk_to_oldk.csv',
                                         test_labels,
                                         challenge='smoking')
    elif sys.argv[1] == 'cohort_selection_challenge':
        # Cohort selection challenge (2018 task 1) -- 13 classes
        train_ids = extract_cohort_status('./datasets/2018_cohort_selection/train')
        test_ids = extract_cohort_status('./datasets/2018_cohort_selection/n2c2-t1_gold_standard_test_data/test')

        chl_train = create_challenge_note('./datasets/n2c2_datasets/train_sentences.txt',
                                          './datasets/n2c2_datasets/train_newk_to_oldk.csv',
                                          train_ids,
                                          challenge='long')
        chl_test = create_challenge_note('./datasets/n2c2_datasets/test_sentences.txt',
                                         './datasets/n2c2_datasets/test_newk_to_oldk.csv',
                                         test_ids,
                                         challenge='long')
    else:
        chl_train, chl_test = [], []

    data_folder = sys.argv[2]
    with open(f'datasets/{data_folder}/train_sentences.txt', 'w') as f:
        wr = csv.writer(f)
        for line in chl_train:
            wr.writerow(line)
    with open(f'datasets/{data_folder}/test_sentences.txt', 'w') as f:
        wr = csv.writer(f)
        for line in chl_test:
            wr.writerow(line)