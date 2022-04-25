# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Add a description here."""

import csv
import os
import datasets
import re

# Dataset description
_DESCRIPTION = """\
Create n2c2 datasets from different challenges for clinical notes redundancy investigation.
Configurations: (1) language_model: pretrained ClinicalBert fine-tuning on MLM and NSP tasks;
(2) smoking_challenge: create dataset to fine-tune ClinicalBERT on the 2006 smoking challenge;
(3) cohort_selection_challenge: 
(4) {WS}r_language_model: create dataset with redundant synthetic notes to pretrain ClinicalBERT.
WS = words percentage and number of sentences;
(5) {WS}r_smoking_challenge: smoking challenge datasets with redundancy;
(6) {WS}r_cohort_selection_challenge: cohort selection challenge datasets with redundancy.
"""

# Link to the official homepage
_HOMEPAGE = "https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/"

# Uncomment the following line to load the dummy version of the data (e.g., for code debugging)
# _FOLDER = {'language_model': './datasets/n2c2_datasets/dummy/language_model/0.0.1/dummy_data',
#            'smoking_challenge': './datasets/2006_smoking_status/dummy/smoking_challenge/0.0.1/dummy_data',
#            'cohort_selection_challenge': './datasets/2018_cohort_selection/dummy/cohort_selection_challenge/0.0.1/dummy_data'}
# for f in os.listdir('./datasets/n2c2_datasets/synthetic_n2c2_datasets'):
#     if os.path.isdir(f) and not re.match(r'_|\.', f):
#         _FOLDER[f'{f}r_language_model'] = f'./datasets/n2c2_datasets/synthetic_n2c2_datasets/{f}/dummy/{f}r_language_model/0.0.1/dummy_data'

_FOLDER = {'language_model': './datasets/n2c2_datasets',
           'smoking_challenge': './datasets/2006_smoking_status',
           'cohort_selection_challenge': './datasets/2018_cohort_selection'}
# Synthetic data for model pretraining
for f in os.listdir(f'{_FOLDER["language_model"]}/synthetic_n2c2_datasets'):
    if os.path.isdir(os.path.join(f'{_FOLDER["language_model"]}/synthetic_n2c2_datasets',
                                  f)) and not re.match(r'_|\.', f):
        _FOLDER[f'{f}r_language_model'] = f'{_FOLDER["language_model"]}/synthetic_n2c2_datasets/{f}'
# Synthetic data for challenges
for f in os.listdir(f'{_FOLDER["smoking_challenge"]}/synthetic_2006_smoking_status'):
    if os.path.isdir(os.path.join(f'{_FOLDER["smoking_challenge"]}/synthetic_2006_smoking_status',
                                  f)) and not re.match(r'_|\.', f):
        _FOLDER[f'{f}r_smoking_challenge'] = f'{_FOLDER["smoking_challenge"]}/synthetic_2006_smoking_status/{f}'
for f in os.listdir(f'{_FOLDER["cohort_selection_challenge"]}/synthetic_2018_cohort_selection'):
    if os.path.isdir(os.path.join(f'{_FOLDER["cohort_selection_challenge"]}/synthetic_2018_cohort_selection',
                                  f)) and not re.match(r'_|\.', f):
        _FOLDER[
            f'{f}r_cohort_selection_challenge'] = f'{_FOLDER["cohort_selection_challenge"]}/synthetic_2018_cohort_selection/{f}'

_SMOKING_LABELS = {'NON-SMOKER': 0,
                   'CURRENT SMOKER': 1,
                   'SMOKER': 2,
                   'PAST SMOKER': 3,
                   'UNKNOWN': 4}

_COHORT_TAGS = ["ABDOMINAL", "ADVANCED-CAD", "ALCOHOL-ABUSE", "ASP-FOR-MI", "CREATININE", "DIETSUPP-2MOS", "DRUG-ABUSE",
                "ENGLISH", "HBA1C", "KETO-1YR", "MAJOR-DIABETES", "MAKES-DECISIONS", "MI-6MOS"]


class N2c2Dataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("0.0.1")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name='language_model', version=VERSION,
                               description="ClinicalBERT pretraining dataset",
                               data_dir='./'),
        datasets.BuilderConfig(name='smoking_challenge', version=VERSION,
                               description="2006 smoking status challenge",
                               data_dir='../2006_smoking_status'),
        datasets.BuilderConfig(name='cohort_selection_challenge', version=VERSION,
                               description="2018 Task 1 cohort selection",
                               data_dir='../2018_cohort_selection')]
    # Redundant builder config
    for f in os.listdir('./datasets/n2c2_datasets/synthetic_n2c2_datasets'):
        if os.path.isdir(os.path.join('./datasets/n2c2_datasets/synthetic_n2c2_datasets',
                                      f)) and not re.match(r'_|\.', f):
            BUILDER_CONFIGS.append(datasets.BuilderConfig(name=f'{f}r_language_model', version=VERSION,
                                                          description=f"ClinicalBERT pretraining redundant dataset, {f}ws",
                                                          data_dir=f'./synthetic_n2c2_datasets/{f}'))
    for f in os.listdir('./datasets/2006_smoking_status/synthetic_2006_smoking_status'):
        if os.path.isdir(os.path.join('./datasets/2006_smoking_status/synthetic_2006_smoking_status',
                                      f)) and not re.match(r'_|\.', f):
            BUILDER_CONFIGS.append(datasets.BuilderConfig(name=f'{f}r_smoking_challenge', version=VERSION,
                                                          description=f"2006 smoking status challenge redundant dataset, {f}ws",
                                                          data_dir=f'../2006_smoking_status/synthetic_2006_smoking_status/{f}'))
    for f in os.listdir('./datasets/2018_cohort_selection/synthetic_2018_cohort_selection'):
        if os.path.isdir(os.path.join('./datasets/2018_cohort_selection/synthetic_2018_cohort_selection',
                                      f)) and not re.match(r'_|\.', f):
            BUILDER_CONFIGS.append(datasets.BuilderConfig(name=f'{f}r_cohort_selection_challenge', version=VERSION,
                                                          description=f"2018 Task 1 cohort selection redundant dataset, {f}ws",
                                                          data_dir=f'../2018_cohort_selection/synthetic_2018_cohort_selection/{f}'))

    DEFAULT_CONFIG_NAME = 'language_model'

    def _info(self):
        if self.config.name == 'language_model' or re.search('r_language_model', self.config.name):
            # TODO: This method specifies the datasets.DatasetInfo object which contains
            #  informations and typings for the dataset
            features = datasets.Features(
                {
                    "sentence": datasets.Value("string"),
                    "document": datasets.Value("string"),
                    "challenge": datasets.Value("string")
                    # These are the features of your dataset like images, labels ...
                }
            )
        elif self.config.name == 'smoking_challenge' or re.search('r_smoking_challenge', self.config.name):
            features = datasets.Features(
                {
                    "note": datasets.Value("string"),
                    "id": datasets.Value("string"),
                    "label": datasets.ClassLabel(5)
                }
            )
        elif self.config.name == 'cohort_selection_challenge' or re.search('r_cohort_selection_challenge',
                                                                           self.config.name):
            features = datasets.Features(
                {
                    "note": datasets.Value("string"),
                    "id": datasets.Value("string"),
                    "label_MET": {tag: datasets.features.ClassLabel(names=["not met", "met"]) for tag in _COHORT_TAGS},
                    "label_NOTMET": {tag: datasets.features.ClassLabel(names=["met", "not met"]) for tag in
                                     _COHORT_TAGS}
                }
            )
        else:
            features = {}
            pass
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting
        #  the data and defining the splits depending on the configuration
        data_dir = _FOLDER[self.config.name]
        # data_dir = dl_manager.download_and_extract(_FOLDER[self.config.name])
        # data_dir = dl_manager.download_and_extract(my_files)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, 'train_sentences.txt'),  # needs to be updated to synthetic
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, 'test_sentences.txt'),
                    "split": "test"
                },
            )
        ]

    def _generate_examples(
            self, filepath, split  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """ Yields examples as (key, example) tuples. """
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.
        with open(filepath, 'r', encoding="utf-8") as f:
            rd = csv.reader(f)
            for id_, row in enumerate(rd):
                if self.config.name == 'language_model' or re.search('r_language_model', self.config.name):
                    if len(row) > 0:
                        yield id_, {
                            "sentence": ','.join(row[2:]),
                            "document": str(row[0]),
                            "challenge": str(row[1]),
                        }
                    else:
                        continue
                elif self.config.name == "smoking_challenge" or re.search('r_smoking_challenge', self.config.name):
                    if len(row) > 0:
                        yield id_, {
                            "note": re.sub('\n', ' ', ','.join(row[2:])),
                            "id": str(row[0]),
                            "label": _SMOKING_LABELS[str(row[1])]
                        }
                elif self.config.name == "cohort_selection_challenge" or re.search('r_cohort_selection_challenge',
                                                                                   self.config.name):
                    if len(row) > 0:
                        tag_lab = {el.split('::')[0]: el.split('::')[1] for el in row[1:-1]}
                        yield id_, {
                            "note": re.sub('\n', ' ', ','.join(row[2:])),
                            "id": str(row[0]),
                            "label_MET": tag_lab,
                            "label_NOTMET": tag_lab
                        }
                else:
                    pass
