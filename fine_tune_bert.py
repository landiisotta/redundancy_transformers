import torch
import torch.nn as nn
from train import train_and_eval
from transformers import BertForPreTraining
from transformers.optimization import get_polynomial_decay_schedule_with_warmup, AdamW
from torch.utils.data import DataLoader
import pickle as pkl
import argparse
import sys
import time

# Run on GPU if available
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CUDA_LAUNCH_BLOCKING = 1


def run_finetuning(checkpoint,
                   data_path,
                   batch_size,
                   n_epochs,
                   learning_rate,
                   patience,
                   sw_redundancy,
                   dev=True):
    """
    Fine-tune ClinicalBERT on MLM and NSP tasks.
    :param checkpoint: ClinicalBERT checkpoint
    :param data_path: path to pkl DatasetDict
    :param batch_size: Batch size
    :param n_epochs: Number of training epochs
    :param learning_rate: Learning rate
    :param patience: (Number of epochs - 1) before stopping
    :param sw_redundancy: number of sentences added and percentage of words randomly replaced for
        redundancy investigation
    :param dev: Whether to perform validation on dev dataset
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    data, tokenizer = pkl.load(open(data_path, 'rb'))

    train, tkn_train = data['train'], tokenizer['train']
    train.set_format(type='torch',
                     columns=['input_ids',
                              'attention_mask',
                              'token_type_ids',
                              'next_sentence_label',
                              'labels'])
    train_loader = DataLoader(train,
                              batch_size=batch_size,
                              shuffle=True)

    if dev:
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

    num_training_steps = len(train_loader.sampler) * n_epochs
    print(f"Number of training steps: {num_training_steps}")
    warmup_steps = round(num_training_steps / 100)
    # warmup_steps = 0
    # Load pretrained model
    model = BertForPreTraining.from_pretrained(checkpoint)
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
    train_and_eval(train_dataloader=train_loader,
                   dev_dataloader=val_loader,
                   model=model,
                   n_epochs=n_epochs,
                   vocab_size=tkn_train.vocab_size,
                   optimizer=optimizer,
                   scheduler=scheduler,
                   patience=patience,
                   sw_redundancy=sw_redundancy)


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
    parser.add_argument('--sw_redundancy',
                        dest='sw_redundancy',
                        type=str,
                        help='Number of sentences added to the end of the note and '
                             'percentage of words randomly replaced for redundancy investigation.')

    config = parser.parse_args(sys.argv[1:])
    start = time.time()
    run_finetuning(checkpoint=config.checkpoint,
                   data_path=config.data_path,
                   n_epochs=config.n_epochs,
                   batch_size=config.batch_size,
                   learning_rate=config.learning_rate,
                   patience=config.patience,
                   dev=config.dev_set,
                   sw_redundancy=config.sw_redundancy)
    print(f"Process finished in {time.time() - start}")
