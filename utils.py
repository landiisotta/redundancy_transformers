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


def _tokenize(sentence, tokenizer_new, tokenizer_old):
    tokenized = []
    new_vocab = set(tokenizer_new.vocab).difference(set(tokenizer_old.vocab))
    for tkn in sentence.split(' '):
        if tkn in new_vocab:
            tokenized.append(tkn)
        else:
            new_tokenized = tokenizer_new.tokenize(tkn)
            tmp = [new_tokenized[0]]
            for t in new_tokenized[1:]:
                if re.match('#', t):
                    tmp.append(t)
                else:
                    tkn_tmp = '##' + t
                    if tkn_tmp in tokenizer_old.vocab:
                        tmp.append(tkn_tmp)
                    else:
                        tmp = tokenizer_old.tokenize(tkn)
                        break
            tokenized.extend(tmp)
    return tokenized
