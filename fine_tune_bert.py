import torch
from train import train_and_eval
from transformers import BertForPreTraining
from transformers.optimization import get_polynomial_decay_schedule_with_warmup, AdamW
from torch.utils.data import DataLoader
import pickle as pkl
import argparse
import sys
import time

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def run_finetuning(checkpoint,
                   data_path,
                   batch_size,
                   n_epochs,
                   learning_rate,
                   num_warmup_steps,
                   num_training_steps,
                   dev=True):
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
                                shuffle=True)
    else:
        val_loader = None

    model = BertForPreTraining.from_pretrained(checkpoint)
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      betas=(0.9, 0.999),
                      eps=1e-6,
                      weight_decay=0.01,
                      correct_bias=False)
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=optimizer,
                                                          num_warmup_steps=num_warmup_steps,
                                                          num_training_steps=num_training_steps,
                                                          lr_end=0.0,
                                                          power=1.0,
                                                          last_epoch=-1)

    train_and_eval(train_loader,
                   val_loader,
                   model,
                   n_epochs=n_epochs,
                   vocab_size=tkn_train.vocab_size,
                   optimizer=optimizer,
                   scheduler=scheduler)


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
    parser.add_argument('--num_warmup_steps',
                        type=int,
                        dest='num_warmup_steps',
                        help='Number of warmup steps for Adam with weight decay optimizer and '
                             'linear learning rate scheduler with warmup')
    parser.add_argument('--num_training_steps',
                        type=int,
                        dest='num_training_steps',
                        help='Total number of training steps')
    parser.add_argument('--dev', dest='dev_set', action='store_true')
    parser.add_argument('--no-dev', dest='dev_set', action='store_false')

    config = parser.parse_args(sys.argv[1:])
    start = time.time()
    run_finetuning(checkpoint=config.checkpoint,
                   data_path=config.data_path,
                   n_epochs=config.n_epochs,
                   batch_size=config.batch_size,
                   learning_rate=config.learning_rate,
                   num_training_steps=config.num_training_steps,
                   num_warmup_steps=config.num_warmup_steps,
                   dev=config.dev_set)
    print(f"Process finished in {time.time() - start}")
