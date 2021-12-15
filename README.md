# Clinical notes redundancy 

## Create dataset
We consider clinical notes from n2c2 datasets for i2b2 challenges. Some notes are shared among tasks hence the 
`create_dataset.py` module combines all unique notes from challenges into unique training and test sets. Edit `utils.py` 
and specify challenge and folder names. Modify input/output directories and output file name in `create_dataset.sh` if needed.
Then run:
 
```
sh create_dataset.sh > out_log.log
```

This module saves training and test notes merging all notes from n2c2 challenges (without duplicates). 
Checkout the `read_notes.ipynb` notebook for a step by step procedure, with checks for duplicates
and comparisons with the literature on the subject [[1-8]](#1).

**Output**: (1) _train|test_n2c2_dataset.txt_ files with (note_id, note_text) on each row;
(2) _train|test_newk_to_oldk.txt_ files with NOTE_ID, CH_ID, CH_NAME with the new-to-old ID correspondence.

## Note preprocessing
The `note_tokenization.py` module preprocesses notes output by the `create_dataset` step via tokenization. It saves them
to a _train|test_sentences.txt_ file in the `datasets/n2c2_datasets` folder that stores a sentence per line. 
Different documents are separated by a blank line.

If, for downstream analyses, we need to use a pretrained tokenizer, this can be added, updated with new words and saved 
locally  by running the script with arguments `pretrained_tokenizer` and `new_pretrained_folder`.

To run the code first modify the required fields in `note_tokenization.sh` then run:

```
sh note_tokenization.sh >> out_log.log
```

## Dataset loading

The script `datasets/n2c2_datasets/n2c2_datasets.py` organizes notes in a `DatasetDict` object, with keys "train|test" and values `Dataset` 
objects with features "sentence", "document", "challenge". The script was tested running the following command in the 
project folder (root of the `datasets` folder):

1. 
```
datasets-cli test datasets/n2c2_datasets --save_infos --all_configs
```  
which return a `dataset_infos.json` file with dataset information

2. 

After modifying file to generate mocked dataset, see 
`<env_path>/lib/python3.9/site-packages/datasets/commands/dummy_data.py`

we run

```
datasets-cli dummy_data datasets/n2c2_datasets \
--auto_generate \
--n_lines=100 \
--match_text_files='train_sentences.txt,test_sentences.txt'
```

to create a dummy version of the data at `datasets/n2c2_datasets/dummy/0.0.1/dummy_data.zip`.

3. 

```
RUN_SLOW=1 pytest tests/test_dataset_common.py::LocalDatasetTest::test_load_real_dataset_n2c2_datasets
RUN_SLOW=1 pytest tests/test_dataset_common.py::LocalDatasetTest::test_load_dataset_all_configs_n2c2_datasets
```

to test both real and dummy datasets.

A cached version of the data will be stored at `~/.cache/huggingface/datasets/n2c2_dataset/default/0.0.1`.  

## Create pretraining data for MLM and NSP with ClinicalBERT

MLM: masked language model

NSP: next sentence prediction

From checkpoint `emilyalsentzer/Bio_ClinicalBERT`

```
sh create_pretraining.sh
```



-------------------------------------------

As a first step we then need to generate the _masked_ and _next sentence prediction_ datasets to fine-tune the
language model with weights initialized by ClinicalBERT. To do that, we run:

```
sh create_pretraining.sh >> out_log.log
``` 

And obtain...




The `note_preprocessing.py` module preprocesses notes in two steps:

```
python3 -m note_preprocessing -dt train|test_n2c2_datasets -ea True
```

The flag `--extract_abbreviation` enables the functions to save a list of abbreviations of the form 

> r' [a-z]{1,3}\. ?[a-z]+\. '

> r' [a-z]{1,3}\. '

with their counts and saves the list to `train|test_n2c2_datasets_abbreviations.csv`. At the same time special 
characters are removed from the text, all text is transformed to lower case, and the header is dropped if it does not 
start with "admission date:". Each note then begins either with a date or the header "admission date:".

If the flag is not present, we run

```
python3 -m note_preprocessing -dt train|test_n2c2_datasets
```

and the functions assume that a file `train|test_n2c2_datasets_abbreviations.csv` with columns 
`ABBRV, COUNT, NEW_ABBRV` exists. The `NEW_ABBRV` column is required to store the new string version without `\.` and uniformed 
throughout different abbreviation configurations of the same words 
(e.g., both "c. diff." and "c. difficile." are replaced by "cdiff"). The text is preprocessed as before with the 
addition of the replacement of abbreviations with their new abbreviation for without `\.`.

In both cases the module outputs the files `train|test_n2c2_datasets_preprocessed.pkl` with the preprocessed notes 
(either w/ or w/o abbreviation replacement). Also, `train|test_n2c2_datasets_sentences_preprocessed.pkl` and 
`train|test_n2c2_datasets_tokenized_preprocessed.pkl` are saved to store the notes in their tokenized version, either
by sentence (list of strings) or by word (list of lists).

## Redundancy detection
The redundancy detection module can be run with three different flags according to the type of redundancy investigation 
desired. 

Training set:
```
python3 -m redundancy_detection -r bn|wn|bp -tr -o bn|wn|bp_redundancy
```

To investigate redundancy on the test set:

```
python3 -m redundancy_detection -r bn|wn|bp -ts -o bn|wn|bp_redundancy
```

The output (both for training and test sets) is the pickle object:

`train|test_bn|wn|bp_redundancy.pkl`

which contains the redundancy information. 

In particular:

> **Between-note (BN) redundancy:** investigates redundancy between note from the same patient 
    (i.e., only longitudinal dataset from the challenges is used). It returns a list of tuples with 
    (note_ID, list of tuples with best aligned pair, aligned sequences1/2, alignment score);

>**Within-note (WN) redundancy:** investigates redundancy within each note counting the occurrences of single sentences.
    It returns a list of named tuples (wn_redundancy) with note_id, score, counts, challenge;

>**Between-patient (BP) redundancy:** investigates between-patient redundancy computing the Levenshtein difference 
    between sentences. It returns a list of named tuples (bp_redundancy) with sentences and score.
     

## Redundancy investigation
Notebook `redundancy_investigation.ipynb` allows to investigate the three redundancies detected for characterization 
and/or removal.


## Synthetic datasets

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