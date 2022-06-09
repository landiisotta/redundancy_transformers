import torch
import torch.nn as nn
from train import train_and_eval
from transformers import BertForPreTraining, AutoTokenizer
from transformers.optimization import get_polynomial_decay_schedule_with_warmup, AdamW
from torch.utils.data import DataLoader
import pickle as pkl
import argparse
import sys
import time
import os
from eval import test
import utils as ut

# Run on GPU if available
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CUDA_LAUNCH_BLOCKING = 1


def run_finetuning(checkpoint,
                   data_path,
                   batch_size,
                   n_epochs,
                   learning_rate,
                   patience,
                   ws_redundancy_train,
                   ws_redundancy_test,
                   dev=True):
    """
    Fine-tune ClinicalBERT on MLM and NSP tasks.
    :param checkpoint: ClinicalBERT checkpoint
    :param data_path: path to pkl DatasetDict
    :param batch_size: Batch size
    :param n_epochs: Number of training epochs
    :param learning_rate: Learning rate
    :param patience: (Number of epochs - 1) before stopping
    :param ws_redundancy_train: number of sentences added and percentage of words randomly replaced for
        redundancy investigation (training set)
    :param ws_redundancy_test: number of sentences added and percentage of words randomly replaced for
        redundancy investigation (test set)
    :param dev: Whether to perform validation on dev dataset
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    if os.path.isdir('./models/pretrained_tokenizer/clinicalBERTmed'):
        print("Using tokenizer updated with medical terms")
        tokenizer_path = './models/pretrained_tokenizer/clinicalBERTmed'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        print("Using original Alsentzer et al. tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(ut.checkpoint)

    # If redundancy thrs do not match between tr and ts it means that we have already pretrained the
    # model on the desired training threshold (and saved the best model) and we can directly test it on the test set.
    if ws_redundancy_train != ws_redundancy_test:
        best_model_dir = f"./runs/BERT-fine-tuning/redu{ws_redundancy_train}tr" \
                         f"{ws_redundancy_train}ts_maxseqlen{data_path.split('maxseqlen')[-1]}"
        data = pkl.load(open(data_path + f'{ws_redundancy_test}.pkl', 'rb'))
        model = BertForPreTraining.from_pretrained(best_model_dir, from_tf=False)
        model.to(DEVICE)
        testset = data['test']
        testset.set_format(type='torch',
                           columns=['input_ids',
                                    'attention_mask',
                                    'token_type_ids',
                                    'next_sentence_label',
                                    'labels'])
        test_loader = DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False)
        out_metrics, _ = test(test_loader, model, len(tokenizer))
        print(f"Model trained on redundancy {ws_redundancy_train}, tested on redundancy {ws_redundancy_test}:")
        print(out_metrics)
        return out_metrics

    data = pkl.load(open(data_path + f'{ws_redundancy_train}.pkl', 'rb'))
    # Load pretrained model
    model = BertForPreTraining.from_pretrained(checkpoint)
    print("Updating Bert with new vocabulary (if extended medical Bert is available)")
    model.resize_token_embeddings(len(tokenizer))

    # train, tkn_train = data['train'].select(range(10)), tokenizer['train']
    train, tkn_train = data['train'], tokenizer
    train.set_format(type='torch',
                     columns=['input_ids',
                              'attention_mask',
                              'token_types_ids',
                              'next_sentence_label',
                              'labels'])
    train_loader = DataLoader(train,
                              batch_size=batch_size,
                              shuffle=True)
    if dev:
        # val = data['test'].select(range(10))
        val = data['test']
        val.set_format(type='torch',
                       columns=['input_ids',
                                'attention_mask',
                                'token_type_ids',
                                'next_sentence_label',
                                'labels'])
        val_loader = DataLoader(val,
                                batch_size=batch_size,
                                shuffle=False)
    else:
        val_loader = None

    num_training_steps = len(train_loader) * n_epochs
    print(f"Number of training steps: {num_training_steps}")
    warmup_steps = round(num_training_steps / 100)
    # warmup_steps = 0
    # Run on multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      betas=(0.9, 0.999),
                      eps=1e-8,
                      weight_decay=0.01,
                      correct_bias=False)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=optimizer,
                                                          num_warmup_steps=warmup_steps,
                                                          num_training_steps=num_training_steps,
                                                          lr_end=0.0,
                                                          power=1.0,
                                                          last_epoch=-1)
    out_metrics = train_and_eval(train_dataloader=train_loader,
                                 dev_dataloader=val_loader,
                                 model=model,
                                 n_epochs=n_epochs,
                                 vocab_size=len(tkn_train),
                                 optimizer=optimizer,
                                 scheduler=scheduler,
                                 patience=patience,
                                 max_seq_len=data_path.split('maxseqlen')[-1],
                                 ws_redundancy_train=ws_redundancy_train,
                                 ws_redundancy_test=ws_redundancy_test)
    return out_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BERT model MLM/NSP fine-tuning")
    parser.add_argument('--checkpoint',
                        type=str,
                        dest='checkpoint',
                        help="Pre-trained checkpoint")
    parser.add_argument('--epochs',
                        type=int,
                        dest='n_epochs',
                        help="Insert number of epochs")
    parser.add_argument('--batch_size',
                        type=int,
                        dest='batch_size',
                        help="Insert batch size")
    parser.add_argument('--data_path',
                        type=str,
                        dest='data_path',
                        help='Insert path to pickled Dataset object')
    parser.add_argument('--learning_rate',
                        type=float,
                        dest='learning_rate',
                        help='Initial learning rate')
    parser.add_argument('--patience',
                        type=int,
                        dest='patience',
                        help='Number of epochs before early stopping.')
    parser.add_argument('--dev', dest='dev_set', action='store_true')
    parser.add_argument('--no-dev', dest='dev_set', action='store_false')
    parser.add_argument('--ws_redundancy_train',
                        dest='ws_redundancy_train',
                        type=str,
                        help='Number of sentences added to the end of the note and '
                             'percentage of words randomly replaced for redundancy investigation.')
    parser.add_argument('--ws_redundancy_test',
                        dest='ws_redundancy_test',
                        type=str,
                        help='Number of sentences added to the end of the note and '
                             'percentage of words randomly replaced for redundancy investigation.')

    config = parser.parse_args(sys.argv[1:])
    start = time.time()
    if os.path.exists('experiments.txt'):
        f = open('experiments.txt', 'a')
    else:
        f = open('experiments.txt', 'w')
        f.write(','.join(['tr_thrs', 'ts_thrs', 'epochs', 'ppl', 'max_seq_len']))
        f.write('\n')
    eval_metrics = run_finetuning(checkpoint=config.checkpoint,
                                  data_path=config.data_path,
                                  n_epochs=config.n_epochs,
                                  batch_size=config.batch_size,
                                  learning_rate=config.learning_rate,
                                  patience=config.patience,
                                  dev=config.dev_set,
                                  ws_redundancy_train=config.ws_redundancy_train,
                                  ws_redundancy_test=config.ws_redundancy_test)
    f.write(','.join([str(config.ws_redundancy_train),
                      str(config.ws_redundancy_test),
                      str(config.n_epochs),
                      str(eval_metrics['ppl']),
                      str(config.data_path.split('maxseqlen')[-1])]))
    f.write('\n')
    f.close()
    print(f"Process finished in {time.time() - start}")
