# Clinical notes redundancy 

> The aim of this work is to investigate the impact of clinical notes redundancy, generated by copy-paste practice, 
> on natural language processing (NLP) models. Given the widespread use of NLP methods in clinical research, it becomes 
> fundamental to understand whether redundancy should be removed from notes agnostically as a preprocessing step or if 
> it can be dealt with on a case-by-case basis depending on the task. Towards this goal, we first estimate the influence 
> of redundancy on language models intrinsically measuring model's performance through perplexity (PPL) when trained on 
> redundant and non-redundant notes and evaluated on real-world clinical text. Secondly, we investigate how redundancy 
> can affect the results of specific NLP tasks (e.g., classification, concept extraction).

## Data
For this project, we considered clinical notes from the _n2c2 NLP research data sets_ for i2b2 challenges. Notes are 
[publicly available](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/) and were released to solve different NLP 
tasks, see dataset description in Table 1.

|Challenge|Year|Task|Data source|Ref|
|---------|----|----|----------|---|
|Smoking status|2006|Classification|Discharge summaries from PH|[[1]](#1)|
|Obesity and comorbidities|2008|Information extraction and classification|Discharge summaries from PH/RPDR|[[2]](#2)|
|Medication extraction|2009|Information extraction|Discharge summaries from PH|[[3]](#3)|
|Concepts, assertions, and relations|2010|Concept extraction|Discharge summaries from PH/BIDMC/UPMC and progress reports from UPMC|[[4]](#4)|
|Coreference resolution|2011|Coreference chain identification|Discharge summaries from PH/BIDMC/UPMC, progress notes from UPMC, clinical and pathology reports from Mayo Clinic, and discharge, radiology, surgical pathology reports, and other from UPMC|[[5]](#5)| 
|Temporal relations|2012|Information extraction|Discharge summaries from PH/BIDMC|[[6]](#6)|
|CAD risk factors|2014|Classification, feature selection|PH EMRs (MGH and BWH)|[[7]](#7)|
|Cohort selection|2018|Classification|Records from 2014 challenge|[[8]](#8)|
|Medication extraction and ADEs|2018|Information extraction, relation classification|Discharge summaries from MIMIC-III|[[9]](#9)|

PH: Partners Healthcare; RPDR: Research Patient Data Repository; BIDMC: Beth Israel Deaconess Medical Center; 
UPMC: University of Pittsburgh Medical Center; MGH: Massachusetts General Hospital; BWH: Brigham and Women's Hospital

**Table 1.** n2c2 datasets description. 

## Pipeline
Implemented modules:
1. `create_dataset`: it takes as input files downloaded from _n2c2 NLP research data sets_ and it combines them into a 
 unique output. All raw-text clinical notes are output in a table format (columns: NOTE_ID, NOTE_TEXT).
2. `note_tokenization`: it takes as input all notes and it tokenizes them at the 
sentence level. It saves tokenized notes to a file with a sentence per line and each note separated by an empty line.
3. `create_pretraining`: it creates the DatasetDict object for BERT fine-tuning. Code modified from 
[ClinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT) and [BERT](https://github.com/google-research/bert). 
4. `fine_tune_bert`: this module fine tunes the pretrained ClinicalBERT model on masked language model and next sequence 
prediction tasks and it evaluates it at each epoch on the test set. 
It returns the best model with early stopping and performances on train and validation (i.e., loss, accuracy, and PPL), 
see `metrics.py`.

### Create n2c2 dataset
We consider the clinical notes from the _n2c2 datasets_ for i2b2 challenges. Some notes are shared among tasks hence the 
`create_dataset.py` module combines all unique notes from challenges into training and test sets. Edit `utils.py` to
specify challenge and folder names. Modify input/output directories and output file name in `create_dataset.sh` if needed.
Then run:
 
```
sh create_dataset.sh
```

**Output**: (1) _train|test_n2c2_dataset.txt_ files with (note_id, note_text) columns;
(2) _train|test_newk_to_oldk.txt_ files with NOTE_ID, CH_ID, CH_NAME columns storing the new-to-old note id 
correspondence. Output folder: `./datasets/n2c2_datasets`.

### Note tokenization
The `note_tokenization.py` module tokenizes the notes generated by the `create_dataset` step. The tokenization process 
happens in two steps:
1. **By sentence**: sentences are defined as (a) delimited by full stop "."; (b) item in list (numeric or bullet point);
2. **At word-level**: we use `spacy en_core_sci_md-0.4.0` model tokenizer and defined a custom tokenizer for special 
tokens. Specifically, we consider as a unique word (a) de-identifiers; (b) dates; (c) times; (d) phone numbers; 
(e) lab/test results; and (f) abbreviations.

**Output**: a _train|test_sentences.txt_ file in the `datasets/n2c2_datasets` output folder that stores a sentence 
per line. Different documents are separated by a blank line. Tokenization at word level can be obtained splitting the 
sentences at " " (space character).

To run the code first modify the required fields in `note_tokenization.sh` then run:

```
sh note_tokenization.sh
```

### Create DatasetDict for BERT pretraining
The `create_pretraining.py` module loads the `datasets.Dataset` object created following the 
[`huggingface` guide](https://huggingface.co/docs/datasets/add_dataset.html), 
see [Dataset loading](#dataset-loading) section.  It outputs a DatasetDict object for BERT model pretraining and saves 
it as `n2c2datasets_forClinicalBERTfinetuning_maxseqlen<N>.pkl` in the output folder .
 
Because we want to intrinsically evaluate the model's performance through PPL, we need to evaluate it on a language 
model task. Hence, for each sentence we replaced the last token (before final [SEP]) with [MASK] during preprocessing. 
This to measure PPL in terms of the ability of the model to predict that last token with only previous words as context. 
Word-level tokenization and sentence length were done according to the sub-word vocabulary used for ClinicalBERT and 
based on the maximum number of words per sentence allowed by the model's configuration (e.g., 128), respectively. 

Run:

```
sh create_pretraining.sh
```

specifying the following hyperparameters
> `max_seq_length`: sequence length;

> `max_predictions_per_seq`: maximum number of [MASK] tokens per sequence;

> `short_seq_prob`: probability of creating sequences shorter than `max_seq_length`;

> `masked_lm_prob`: percentage of token positions randomly selected per sentence;

> `dupe_factor`:  number of times data should be duplicated with different masks.

### Fine-tuning pretrained ClinicalBERT
The `fine_tune_bert.py` module takes as input the `DatasetDict` object with clinical notes preprocessed for masked 
language model (MLM) and next sequence prediction (NSP) tasks and it outputs the best model configuration after training.
The module combines `train.py`, `test.py`, and `metrics.py` to train the 
[pretrained ClinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT/tree/master/lm_pretraining) on our datasets and 
evaluate its performance on the validation set in terms of PPL, computed as the exponential of the mean cross-entropy 
loss when predicting the last word of each sentence.

Run:

```
sh fine_tune_bert.sh
```

Hyperparameters:

> `epochs`: number of complete training passes;

> `batch_size`: number of samples to process (e.g., 256, 1024 from the BERT paper);

> `learning_rate:` learning rate for gradient descent (e.g.,  1e-4 from the BERT paper);

> `num_training_steps`: total number of training steps (i.e., number of sentences * epochs);

> `num_warmup_step`: number of warmup steps for AdamW optimizer with linear learning rate decay with warmup, 
it should be 1% of the total training steps;

> `patience`: (number of epochs - 1) before early stopping.

**Remarks:** BERT experiments run 1024*128 or 256*512 tokens/batch on 3.3B words. They trained the model for 40 epochs, 
which correspond to ~1M training steps (epochs*batches). Their warmup steps are 10000, i.e., 1% of the training steps, 
and half of one epoch, which includes 25177 batches.

For our experiment we have 256*128 tokens/batch, which correspond to ~900 batches. For a warmup steps of 400, we have to 
fine-tune our MLM for 40,000 steps, i.e., 45 epochs 
(although we include the early stopping with patience 5 to avoid overfitting).

## Dataset loading
The module `datasets/n2c2_datasets/n2c2_datasets` prepares the input Dataset object for fine-tuning the pretrained BERT 
model (i.e., ClinicalBERT) on the MLM and NSP tasks. It was implemented based on the 
[`huggingface` guide](https://huggingface.co/docs/datasets/add_dataset.html).

The script `datasets/n2c2_datasets/n2c2_datasets.py` organizes notes in a `DatasetDict` object, with keys "train|test" 
and values `Dataset` 
objects with features "sentence", "document", "challenge". The script was tested running the following command in the 
project folder (root of the `datasets` folder):

#### Step 1
```
datasets-cli test datasets/n2c2_datasets --save_infos --all_configs
```  
which return a `dataset_infos.json` file with dataset information.

#### Step 2
In order to run the second test and generate the dummy dataset, we needed to modify the file at 
`<env_path>/lib/python3.9/site-packages/datasets/commands/dummy_data.py`

Then we run
```
datasets-cli dummy_data datasets/n2c2_datasets \
--auto_generate \
--n_lines=100 \
--match_text_files='train_sentences.txt,test_sentences.txt'
```
to create a dummy version of the data at `datasets/n2c2_datasets/dummy/0.0.1/dummy_data.zip`.

#### Step 3
```
RUN_SLOW=1 pytest tests/test_dataset_common.py::LocalDatasetTest::test_load_real_dataset_n2c2_datasets
RUN_SLOW=1 pytest tests/test_dataset_common.py::LocalDatasetTest::test_load_dataset_all_configs_n2c2_datasets
```
to test both real and dummy datasets.

A cached version of the data will be stored at `~/.cache/huggingface/datasets/n2c2_dataset/default/0.0.1`.  

**Remark**: this loading dataset script can be edited to add dataset configurations other than the "language model", 
e.g., for specific tasks.

# References
<a id="1">[1]</a> Uzuner, Ö., Goldstein, I., Luo, Y., & Kohane, I. (2008). Identifying patient smoking status from medical 
discharge records. _Journal of the American Medical Informatics Association_, 15(1), 14-24.  

<a id="2">[2]</a> Uzuner, Ö. (2009). Recognizing obesity and comorbidities in sparse data. 
_Journal of the American Medical Informatics Association_, 16(4), 561-570.

<a id="3">[3]</a> Uzuner, Ö., Solti, I., & Cadag, E. (2010). 
Extracting medication information from clinical text. _Journal of the American Medical Informatics Association_, 17(5), 514-518.

<a id="4">[4]</a> Uzuner, Ö., South, B. R., Shen, S., & DuVall, S. L. (2011). 2010 i2b2/VA challenge on concepts, 
assertions, and relations in clinical text. _Journal of the American Medical Informatics Association_, 18(5), 552-556.

<a id="5">[5]</a> Uzuner, O., Bodnari, A., Shen, S., Forbush, T., Pestian, J., & South, B. R. (2012). 
Evaluating the state of the art in coreference resolution for electronic medical records. 
_Journal of the American Medical Informatics Association_, 19(5), 786-791.

<a id="6">[6]</a> Sun, W., Rumshisky, A., & Uzuner, O. (2013). Evaluating temporal relations in clinical text: 
2012 i2b2 Challenge. _Journal of the American Medical Informatics Association_, 20(5), 806-813.

<a id="7">[7]</a> Kumar, V., Stubbs, A., Shaw, S., & Uzuner, Ö. (2015). 
Creation of a new longitudinal corpus of clinical narratives. _Journal of biomedical informatics_, 58, S6-S10.

<a id="8">[8]</a> Stubbs, A., Filannino, M., Soysal, E., Henry, S., & Uzuner, Ö. (2019). Cohort selection for clinical 
trials: n2c2 2018 shared task track 1. _Journal of the American Medical Informatics Association_, 26(11), 1163-1171.

<a id="9">[9]</a> Henry, S., Buchan, K., Filannino, M., Stubbs, A., & Uzuner, O. (2020). 2018 n2c2 shared task on 
adverse drug events and medication extraction in electronic health records. 
_Journal of the American Medical Informatics Association_, 27(1), 3-12.