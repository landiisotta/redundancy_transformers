# n2c2 challenge datasets abbreviations
chll = ['smoking', 'obesity',
        'med_extraction', 'cpt_extraction',
        'temp_rel', 'long',
        'med_extraction_tsk2']

# Folders where files are stores. Subfolder and file names were preserved.
# Main folder names were changed respect to those downloaded from
# https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
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
# checkpoint = "emilyalsentzer/Bio_ClinicalBERT"
checkpoint = "./models/pretrained_tokenizer/clinicalBERT"

# Loss function

# Needleman-Wunsch alignment weights
match = 2
mismatch = -10
gap_open = -0.5
gap_extend = -0.1

# Minimum tokens in sentence
min_sen_len = 1
# Minimum sentence counts for BP redundancy
min_sen_count = 5
