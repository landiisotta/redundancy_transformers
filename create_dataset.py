import xml.etree.ElementTree as ElT
import os
import csv
import re
import utils as ut
import argparse
import sys
import time


def read_txt(file):
    """
    Txt file format reader (first line corresponds to the record number if any)
    :param file: file name
    :return: note
    :rtype: str
    """
    with open(file, mode='r', encoding='utf-8-sig') as f:
        text = f.read().rstrip()
    return text


def readxml(file):
    """
    XML file format reader.
    :param file: file name
    :return: note
    :rtype: str
    """

    notes = []

    tree = ElT.parse(file)
    root = tree.getroot()
    n_id, text = None, None
    for elem in root.iter():
        # If RECORD|doc element is present
        # use it as note_id, else set to None
        if elem.tag == 'RECORD' or elem.tag == 'doc':
            try:
                n_id = elem.attrib['ID']
            except KeyError:
                n_id = elem.attrib['id']
        # Element TEXT|text stores the clinical note
        if re.match(r'[Tt][Ee][Xx][Tt]', elem.tag):
            text = elem.text
            notes.append([n_id, text])
    return notes


def create_dt(challenge, data_folder, train=True):
    """
    Create challenge datasets. Read notes from files from
    different challenges. Files with notes were downloaded from
    https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
    preserving all sub_folder/file names. Challenge folders organization is reported in utils.
    :param challenge: challenge name (see utils.py)
    :param data_folder: path to the datasets folder
    :param train: bool, if True function returns the training set,
        test set otherwise (default True)
    :return: all notes combined (w/o duplicates)
    :rtype: dict
    """
    # Create file paths
    if train:
        file_path = ut.train_files[challenge]
    else:
        file_path = ut.test_files[challenge]
    if not isinstance(file_path, list):
        file_path = [data_folder + file_path]
    else:
        file_path = [data_folder + ff for ff in file_path]

    files = []
    myparser = {}
    for fld in file_path:
        if (challenge == 'temp_rel' and train) or challenge == 'med_extraction_tsk2':
            files.extend([os.path.join(fld, ff) for ff in os.listdir(fld) if re.search(r'\.txt', ff)])
        elif challenge in ['smoking', 'obesity']:
            if train:
                files.extend([os.path.join(fld, ff) for ff in os.listdir(fld)
                              if re.search(r'records_training|train_all', ff)])
            else:
                files.extend([os.path.join(fld, ff) for ff in os.listdir(fld)
                              if re.search(r'records_test|test_all_groundtruth', ff)])
        else:
            files.extend([os.path.join(fld, ff) for ff in os.listdir(fld) if not re.match(r'^\.', ff)])
    # Extract note IDs
    if challenge in ['med_extraction', 'temp_rel', 'med_extraction_tsk2']:
        for ff in files:
            text = read_txt(ff)
            if challenge != 'temp_rel':
                idx = ff.split('/')[-1].split('.')[0]
            else:
                idx = re.search(r'[0-9]+', ff.split('/')[-1]).group(0)
            myparser[idx] = text
    elif challenge == 'cpt_extraction':
        for ff in files:
            text = read_txt(ff)
            if re.match(r" *A[Dd][Mm][Ii][Ss][Ss][Ii][Oo][Nn] D[Aa][Tt][Ee] *:?", text):
                idx = re.search(r'[0-9]+(_[a-z])*', ff.split('/')[-1]).group(0)
            else:
                ma = re.match(r'[0-9]+(_[a-z])*', text)
                idx = ma.group(0)
                tt_idx = ma.span()[1]
                text = text[tt_idx:]
            myparser[idx] = text
    elif challenge in ['smoking', 'obesity']:
        for ff in files:
            out = readxml(ff)
            myparser = {elem[0]: elem[1] for elem in out}
            return myparser
    else:
        for ff in files:
            text = readxml(ff)[0][1]
            idxa = re.search('[0-9]+-[0-9]+', ff.split('/')[-1]).group(0)
            idx = idxa.split('-')[0]
            n = idxa.split('-')[1]
            long_idx = '::'.join([idx, n])
            myparser[long_idx] = text
    return myparser


"""
Private functions
"""


def _merge_dict(dict_list):
    """
    Function that merges challenge dictionaries into one. Duplicated keys are renamed with progressive
    numbers. The function outputs a new merged dictionary and a dictionary with keys: (old_key, challenge id)
    to keep track of the notes for downstream tasks.
    :param dict_list: list of tuples (challenge id, dictionary with notes)
    :return: (new dictionary, new_key to old_key dictionary)
    :rtype: tuple
    """
    dt = {}
    newk_to_oldk = {}
    count = {}
    for ch_name, dd in dict_list:
        for k, note in dd.items():
            if k in count:
                count[k] += 1
            else:
                count[k] = 0
            if k in dt:
                i = count[k]
                new_k = '_'.join([k, str(i)])
                dt[new_k] = note
                newk_to_oldk[new_k] = (k, ch_name)
            else:
                dt[k] = note
                newk_to_oldk[k] = (k, ch_name)
    return dt, newk_to_oldk


def _write_key_dict(name, k_dict, output_folder, train=True):
    """
    Write new_to_old_key dictionary to file.
    :param name: output file name
    :param k_dict: dictionary with link between new note ids and old note ids per challenge
    :param output_folder:
    """
    with open(os.path.join(output_folder, f'{name}.csv'), 'w') as f:
        wr = csv.writer(f)
        # NOTE_ID: new ID from merge;
        # CH_ID: challenge old ID;
        # CH_NAME: challenge name.
        wr.writerow(["NOTE_ID", "CH_ID", "CH_NAME"])
        for k, tup in k_dict.items():
            wr.writerow([k, tup[0], tup[1]])


def _write_file(dataset, file_name, output_folder, train=True):
    """
    Function that saves notes to .txt file
    :param dataset: {note_id: text}
    :type dataset: dict
    :param file_name: output file name
    :type file_name: str
    :param train: whether to save the train or test sets
    :type train: bool
    """
    partition = 'train'
    if not train:
        partition = 'test'
    with open(os.path.join(output_folder, f'{partition}_' + file_name + '.txt'), 'w') as f:
        wr = csv.writer(f)
        for nid, text in dataset.items():
            wr.writerow([nid, text])


# Main
if __name__ == '__main__':
    start = time.process_time()
    print("Creating N2C2 merged dataset:")
    parser = argparse.ArgumentParser(description='Create unique dataset (train/test) from n2c2 challenges.')
    parser.add_argument('-i',
                        '--input',
                        type=str,
                        dest='input_folder',
                        help='Input folder')
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        dest='output_folder',
                        help='Output folder')
    parser.add_argument('-of',
                        '--outputfile',
                        type=str,
                        dest='output_file',
                        help='Output file name')
    config = parser.parse_args(sys.argv[1:])

    train_list = []
    test_list = []
    for chll in ut.chll:
        tr = create_dt(chll, config.input_folder)
        if chll == 'med_extraction':
            ts_tmp = create_dt(chll, config.input_folder, train=False)
            ts = {k: note for k, note in ts_tmp.items() if k not in tr.keys()}
        else:
            ts = create_dt(chll, config.input_folder, train=False)
        train_list.append((chll, tr))
        test_list.append((chll, ts))
    # Create unique dataset
    dt_train, tr_k_dict = _merge_dict(train_list)
    dt_test, ts_k_dict = _merge_dict(test_list)

    # Check for possible duplicates:
    # Same id and same challenge between train/test
    inter = set(tr_k_dict.values()).intersection(ts_k_dict.values())
    # Same note between train and test
    dup = []
    for k, n in dt_train.items():
        for kt, nt in dt_test.items():
            if n == nt:
                dup.append(k)
    # Remove duplicates from training set
    dt_train_rid = {k: val for k, val in dt_train.items() if tr_k_dict[k] != list(inter)[0] and k not in dup}
    tr_k_dict_rid = {k: val for k, val in tr_k_dict.items() if tr_k_dict[k] != list(inter)[0] and k not in dup}
    print(f"Original training set number of notes: {len(dt_train)}")
    print(f"{len(dt_train) - len(dt_train_rid)} duplicated notes (w/ test set) dropped")
    print(f"Training set number of notes: {len(dt_train_rid)}")
    print(f"Test set number of notes: {len(dt_test)}")

    _write_file(dt_train_rid,
                file_name=config.output_file,
                output_folder=config.output_folder)
    _write_key_dict('train_newk_to_oldk', tr_k_dict_rid,
                    output_folder=config.output_folder, train=False)

    _write_file(dt_test, file_name=config.output_file,
                output_folder=config.output_folder,
                train=False)
    _write_key_dict('test_newk_to_oldk', ts_k_dict,
                    output_folder=config.output_folder)

    print(f"Train/test notes and dictionary keys saved")
    print(f"Task ended in {round(time.process_time() - start, 2)}s.\n")
