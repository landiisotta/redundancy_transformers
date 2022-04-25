import csv
import re

# n2c2 challenge datasets folder names
chll = ['smoking',
        'obesity',
        'med_extraction',
        'cpt_extraction',
        'temp_rel',
        'long',
        'med_extraction_tsk2']

FOLDER = {'language_model': './datasets/n2c2_datasets',
          'smoking_challenge': './datasets/2006_smoking_status'}

# Folders where files are stores. Sub folder and file names were preserved.
# Main folder names were changed
train_files = {'smoking': '/2006_smoking_status',
               'obesity': '/2008_obesity',
               'med_extraction': ['/'.join(['/2009_medication/training.sets.released',
                                            str(i + 1)]) for i in range(10)],
               'cpt_extraction': ['/2010_relations/concept_assertion_relation_training_data/beth/txt',
                                  '/2010_relations/concept_assertion_relation_training_data/partners/txt',
                                  '/2010_relations/concept_assertion_relation_training_data/partners/unannotated'],
               'temp_rel': ['/2012_temporal_relations/2012-06-18.release-fix',
                            '/2012_temporal_relations/2012-07-06.release-fix'],
               'long': ['/2014_heart_disease/training-RiskFactors-Complete-Set1/',
                        '/2014_heart_disease/training-RiskFactors-Complete-Set2/'],
               'med_extraction_tsk2': '/2018_medication_extraction/training_20180910/'}

test_files = {'smoking': '/2006_smoking_status',
              'obesity': '/2008_obesity',
              'med_extraction': '/2009_medication/train.test.released.8.17.09/',
              'cpt_extraction': '/2010_relations/test_data',
              'temp_rel': '/2012_temporal_relations/2012-08-06.test-data-release/txt',
              'long': '/2014_heart_disease/testing-RiskFactors-Complete/',
              'med_extraction_tsk2': '/2018_medication_extraction/test'}

# Pre-trained checkpoint
checkpoint = "./models/pretrained_tokenizer/clinicalBERT"
checkpointmod = "./models/pretrained_tokenizer/clinicalBERTmod"


def extract_words_to_add():
    """
    Function that enriches the BERT vocabulary with words selected for the synthetic simulations
    and special numeric tokens from the notes
    :return: set of tokens to add to the Bert vocabulary
    """
    # Words used for synthetic replacement
    fullw_set = set()
    with open('./datasets/n2c2_datasets/synthetic_n2c2_datasets/test_w_to_idx.txt') as f:
        rd = csv.reader(f)
        for r in rd:
            fullw_set.add(r[0])
    with open('./datasets/n2c2_datasets/synthetic_n2c2_datasets/train_w_to_idx.txt') as f:
        rd = csv.reader(f)
        for r in rd:
            if r[0] not in fullw_set:
                fullw_set.add(r[0])
    # Build clinical notes vocabulary of "special characters"
    special_set = set()
    with open('./datasets/n2c2_datasets/train_sentences.txt') as f:
        rd = csv.reader(f)
        for r in rd:
            tkns = r[-1].split(' ')
            special_set.update(tkns)
    with open('./datasets/n2c2_datasets/test_sentences.txt') as f:
        rd = csv.reader(f)
        for r in rd:
            tkns = r[-1].split(' ')
            special_set.update(tkns)
    notes_vocab = set([w for w in special_set if re.match(r'[0-9]+', w)])
    add_tokens = fullw_set.union(notes_vocab)
    return add_tokens
