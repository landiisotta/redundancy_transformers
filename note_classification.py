import torch
from tqdm.auto import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers.optimization import get_polynomial_decay_schedule_with_warmup, AdamW, get_scheduler
import sys
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import pickle as pkl
import metrics
import time
import csv
import os
from models.multilabel_bert import MultiLabelBertTask

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
GPUS = max(torch.cuda.device_count(), 1)


# Smoking challenge
# Measures to evaluate classification: Precision, Recall, F1 score
# Evaluation on each smoking category separately, then microaverage and macroaverage (for all metrics)
def training(train_set,
             dev_set,
             model,
             optimizer,
             lr_scheduler,
             batches,
             weighting,
             challenge):
    """
    Training/validation step.

    :param train_set:
    :param dev_set:
    :param model:
    :param optimizer:
    :param lr_scheduler:
    :param batches:
    :param weighting:
    :param challenge:
    :return: training and validation loss
    """
    # Training
    model.train()
    loss_batches = 0
    train_metrics = metrics.TaskMetrics(challenge=challenge)
    for batch in train_set:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)

        wloss, _ = _compute_wloss(batch, outputs, batches, weighting, challenge=challenge)
        wloss.sum().backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        if weighting:
            loss_batches += wloss.sum().item()
        else:
            if batches:
                loss_batches += wloss.sum().item() * batch['input_ids'].shape[0]
            else:
                loss_batches += wloss.sum().item()

    # Validation
    model.eval()
    loss_batches_eval = 0
    for batch in dev_set:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        val_loss, output = _compute_wloss(batch, outputs, batches, weighting, challenge)

        if weighting:
            loss_batches_eval += val_loss.sum().item()
            train_metrics.add_batch(batch['labels'][0].item(),
                                    torch.argmax(output).item())
        else:
            if batches:
                loss_batches_eval += val_loss.sum().item() * batch['input_ids'].shape[0]
                # extend
                if challenge == 'smoking_challenge':
                    train_metrics.add_batch(batch['labels'].view(-1).tolist(),
                                            torch.argmax(outputs.logits, dim=-1).tolist())
                elif challenge == 'cohort_selection_challenge':
                    train_metrics.add_batch(batch['labels'].tolist(),
                                            _get_labels(output))
            else:
                loss_batches_eval += val_loss.sum().item()
                if challenge == 'smoking_challenge':
                    train_metrics.add_batch(batch['labels'][0].item(),
                                            torch.argmax(output).item())
                elif challenge == 'cohort_selection_challenge':
                    train_metrics.add_batch([batch['labels'][0].tolist()],
                                            _get_labels(output))

    return loss_batches / (len(train_set.sampler) * GPUS), loss_batches_eval / (
            len(dev_set.sampler) * GPUS), train_metrics.compute()


def _normal_density(x, mu, sigma):
    """
    Compute normal density function from z-scores
    """
    return (2. * np.pi * sigma ** 2.) ** -.5 * np.exp(-.5 * (x - mu) ** 2. / sigma ** 2.)


def _collate_fn(batch):
    """
    Custom collate function for DataLoader.
    """
    input_ids = batch['input_ids']
    labels = batch['labels']
    # Added multi-label case
    if isinstance(labels[0], list):
        return {'input_ids': torch.tensor(input_ids),
                'labels': torch.tensor(labels)}
    else:
        return {'input_ids': torch.tensor(input_ids),
                'labels': torch.tensor([[lab] for lab in labels])}


def _collate_fn_batch(batch):
    """
    Custom collate function for DataLoader.
    """
    input_ids, labels = [], []
    for b in batch:
        input_ids.append(b['input_ids'][0])
        labels.append(b['labels'][0])
    # Added multi-label case
    if isinstance(labels[0], list):
        return {'input_ids': torch.tensor(input_ids),
                'labels': torch.tensor([lab for lab in labels])}
    else:
        return {'input_ids': torch.tensor(input_ids),
                'labels': torch.tensor([[lab] for lab in labels])}


def _compute_wloss(batch, outputs, batches, weighting_method, challenge):
    """
    If weighting method is enabled: compute loss of overlapping note chunks weighting more the initial vectors;
     Otherwise: return batched loss.
    """
    if weighting_method:
        # weight initial vectors more
        mid = float(int(batch['input_ids'].shape[0] / 2))
        if mid > 0:
            norm_dist = np.array(
                [(i / mid) for i in range(batch['input_ids'].shape[0])])
            weights = torch.tensor(np.array([_normal_density(norm_dist, 0, 1)]), device=DEVICE).double()
        else:
            weights = torch.tensor(np.array([[1]]), device=DEVICE).double()

        wloss = nn.functional.cross_entropy(torch.matmul(weights, outputs.logits),
                                            batch['labels'][0].view(-1))
        output = torch.matmul(weights, outputs.logits)
    else:
        if batches:
            if challenge == 'smoking_challenge':
                wloss = outputs.loss
                output = outputs.logits
            elif challenge == 'cohort_selection_challenge':
                wloss = nn.functional.binary_cross_entropy(outputs, batch['labels'].float())
                output = outputs
            else:
                output, wloss = None, None
        else:
            if challenge == 'smoking_challenge':
                weights = torch.tensor(np.array([[1] * batch['labels'].shape[0]]), device=DEVICE).double()
                wloss = nn.functional.cross_entropy(torch.matmul(weights, outputs.logits),
                                                    batch['labels'][0].view(-1))
                output = torch.matmul(weights, outputs.logits)
            elif challenge == 'cohort_selection_challenge':
                output = torch.max(outputs, dim=0).values.view(1, -1)
                wloss = nn.functional.binary_cross_entropy(output,
                                                           batch['labels'][0].view(1, -1).float())
            else:
                output, wloss = None, None
    return wloss, output


def _get_labels(out_sigmoids, threshold=0.5):
    return np.array(out_sigmoids > threshold, dtype=int).tolist()


if __name__ == '__main__':
    parser = ArgumentParser(description='Note classification task')
    parser.add_argument('--dataset',
                        type=str,
                        dest='dataset',
                        help='Path to dataset splits')
    parser.add_argument('--checkpoint',
                        type=str,
                        dest='checkpoint',
                        help='Pre-trained model checkpoint')
    parser.add_argument('--epochs',
                        type=int,
                        dest='epochs',
                        help='Number of epochs')
    parser.add_argument('--learning_rate',
                        type=float,
                        help='Learning rate',
                        dest='learning_rate')  # 5e-5
    parser.add_argument('--n_classes',
                        type=int,
                        help='Number of classes or number of labels for '
                             'the multi-label binary classification task',
                        dest='n_classes')
    parser.add_argument('--batch_size',
                        type=int,
                        help='If non-overlapping notes N batches',
                        dest='batch_size',
                        default=None)
    parser.add_argument('--challenge',
                        type=str,
                        help='Challenge name',
                        dest='challenge')
    parser.add_argument('--weighting',
                        help='Enabling weighting strategy for overlapping note chunks',
                        dest='weighting',
                        action='store_true')
    parser.add_argument('--no-weighting',
                        help='Disable weighting strategy',
                        dest='weighting',
                        action='store_false')

    start = time.process_time()

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    config = parser.parse_args(sys.argv[1:])

    if config.challenge == 'cohort_selection_challenge' and config.weighting:
        print("Incompatible commands, cannot fine-tune cohort selection challenge with weighting method.")
        print("Exiting main...")
        sys.exit()

    writer_train = SummaryWriter(f'./runs/BERT-task-{config.challenge}/tensorboard/train/{config.learning_rate}')
    writer_val = SummaryWriter(f'./runs/BERT-task-{config.challenge}/tensorboard/validation/{config.learning_rate}')
    writer_test = SummaryWriter(f'./runs/BERT-task-{config.challenge}/tensorboard/test/{config.learning_rate}')

    if config.challenge == 'smoking_challenge':
        model = AutoModelForSequenceClassification.from_pretrained(config.checkpoint,
                                                                   num_labels=config.n_classes)
        model = model.double()
    elif config.challenge == 'cohort_selection_challenge':
        model = MultiLabelBertTask(checkpoint=config.checkpoint,
                                   labels=config.n_classes)
    else:
        sys.exit()

    ws = config.dataset.split('ws')[-1].split('.')[0]
    seqlen = config.dataset.split('maxlen')[-1].split('ws')[0]
    if config.challenge == 'cohort_selection_challenge':
        class_label = config.dataset.split('_')[-2]
    else:
        class_label = None
    data = pkl.load(open(config.dataset, 'rb'))

    train, val, test = data['train'], data['validation'], data['test']

    if config.batch_size:
        if config.weighting:
            print("Weighting strategy is incompatible with batching, disabling it.")
            config.weighting = False
        if ws != 'None':
            print(f"Collating data into batches of size {config.batch_size}, please note that "
                  f"Dataset configuration has window enabled at {ws}, this will cut the note at the length defined"
                  f" {seqlen}. No weighting method applied.")
        else:
            print(f"Collating data into batches of size {config.batch_size}.")
        train_loader = DataLoader(train,
                                  shuffle=True,
                                  collate_fn=_collate_fn_batch,
                                  batch_size=config.batch_size)
        val_loader = DataLoader(val,
                                shuffle=False,
                                collate_fn=_collate_fn_batch,
                                batch_size=config.batch_size)
        test_loader = DataLoader(test,
                                 shuffle=False,
                                 collate_fn=_collate_fn_batch,
                                 batch_size=config.batch_size)
    else:
        if ws == 'None':
            print("Window is not present, but because batch size has not been specified, experiment"
                  f" will be run with batch size = 1.")
            if config.weighting:
                print("Weighting strategy cannot be applied with batch size = 1. Disabling it.")
                config.weighting = False
        train_loader = DataLoader(train,
                                  collate_fn=_collate_fn,
                                  shuffle=True,
                                  batch_size=None,
                                  batch_sampler=None)
        val_loader = DataLoader(val,
                                collate_fn=_collate_fn,
                                shuffle=False,
                                batch_size=None,
                                batch_sampler=None)
        test_loader = DataLoader(test,
                                 collate_fn=_collate_fn,
                                 shuffle=False,
                                 batch_size=None,
                                 batch_sampler=None)

    num_training_steps = config.epochs * len(train_loader)
    warmup_steps = round(num_training_steps / 100)
    print(f"Training steps: {num_training_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Learning rate: {config.learning_rate}\n")

    progress_bar = tqdm(range(num_training_steps))

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    # lr_scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=optimizer,
    #                                                          num_warmup_steps=warmup_steps,
    #                                                          num_training_steps=num_training_steps,
    #                                                          lr_end=0.0,
    #                                                          power=1.0,
    #                                                          last_epoch=-1)
    lr_scheduler = get_scheduler('linear',
                                 num_warmup_steps=warmup_steps,
                                 num_training_steps=num_training_steps,
                                 optimizer=optimizer)

    # Run on multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    val_split = round(val.num_rows / train.num_rows, 1)
    if not os.path.isdir(f'./runs/BERT-task-{config.challenge}/experiments'):
        os.makedirs(f'./runs/BERT-task-{config.challenge}/experiments')
    if config.challenge == 'cohort_selection_challenge':
        metrics_file = open(os.path.join(f'./runs/BERT-task-{config.challenge}/experiments/',
                                         f'metrics_valsplit{val_split}lr{config.learning_rate}ws{ws}seqlen{seqlen}batch'
                                         f'{config.batch_size}weights{config.weighting}_{class_label}.csv'), 'w')
    else:
        metrics_file = open(os.path.join(f'./runs/BERT-task-{config.challenge}/experiments/',
                                         f'metrics_valsplit{val_split}lr{config.learning_rate}ws{ws}seqlen{seqlen}batch'
                                         f'{config.batch_size}weights{config.weighting}.csv'), 'w')
    wr = csv.writer(metrics_file)
    colnames = ['learning_rate', 'val_split', 'window_size', 'sequence_length', 'batch_size', 'weighting_enabled',
                'epoch', 'tr_loss', 'val_loss'] + ['f1_micro', 'f1_macro'] + \
               [f'f1_class{cl}' for cl in range(config.n_classes)] + ['p_micro', 'p_macro'] + \
               [f'p_class{cl}' for cl in range(config.n_classes)] + ['r_micro', 'r_macro'] + \
               [f'r_class{cl}' for cl in range(config.n_classes)]
    wr.writerow(colnames)
    fixed_info = [config.learning_rate, val_split, ws, seqlen, config.batch_size, config.weighting]

    model.to(DEVICE)
    # Training
    for epoch in range(config.epochs):
        tr_loss, val_loss, val_metrics = training(train_loader,
                                                  val_loader,
                                                  model,
                                                  optimizer,
                                                  lr_scheduler,
                                                  config.batch_size,
                                                  config.weighting,
                                                  config.challenge)
        line = fixed_info + [epoch, tr_loss, val_loss]
        writer_train.add_scalar('epoch_loss', tr_loss, epoch)
        writer_val.add_scalar('epoch_loss', val_loss, epoch)
        for k in val_metrics.keys():
            for kk, score in val_metrics[k].items():
                writer_val.add_scalar(f'Validation {kk}', score, epoch)
                line.append(score)
        if epoch % 10 == 0 or epoch == (config.epochs - 1):
            print(f"Epoch {epoch} -- Training loss {round(tr_loss, 4)}")
            print(f"Epoch {epoch} -- Validation loss {round(val_loss, 4)}")
            print('\n')
            print(f"Epoch {epoch} -- Classification metrics:")
            for k in val_metrics.keys():
                print(k)
                for kk, val in val_metrics[k].items():
                    print(f"{kk}: {val}")
                print('\n')
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': tr_loss}, f'./runs/BERT-task-{config.challenge}/checkpoint.pt')
        wr.writerow(line)
    # Test
    line = fixed_info + ['', '', '']
    model.eval()
    test_metrics = metrics.TaskMetrics(challenge=config.challenge)
    test_loss = 0
    for test_batch in test_loader:
        test_batch = {k: v.to(DEVICE) for k, v in test_batch.items()}
        with torch.no_grad():
            outputs = model(**test_batch)

        wloss, output = _compute_wloss(test_batch,
                                       outputs,
                                       config.batch_size,
                                       weighting_method=config.weighting,
                                       challenge=config.challenge)
        if config.weighting:
            test_metrics.add_batch(test_batch['labels'][0].item(), torch.argmax(output).item())
            test_loss += wloss.sum().item()
        else:
            if config.batch_size:
                test_loss += wloss.sum().item() * test_batch['input_ids'].shape[0]
                # extend
                if config.challenge == 'smoking_challenge':
                    test_metrics.add_batch(test_batch['labels'].view(-1).tolist(),
                                           torch.argmax(outputs.logits, dim=-1).tolist())
                elif config.challenge == 'cohort_selection_challenge':
                    test_metrics.add_batch(test_batch['labels'].tolist(),
                                           _get_labels(output))

            else:
                test_loss += wloss.sum().item()
                if config.challenge == 'smoking_challenge':
                    test_metrics.add_batch(test_batch['labels'][0].item(), torch.argmax(output).item())
                elif config.challenge == 'cohort_selection_challenge':
                    test_metrics.add_batch([test_batch['labels'][0].tolist()], _get_labels(output))

    eval_metrics = test_metrics.compute()
    writer_test.add_scalar('Test loss', test_loss / (len(test_loader) * GPUS))

    for k in eval_metrics.keys():
        for kk, score in eval_metrics[k].items():
            writer_test.add_scalar(f'Test {kk}', score)
            line.append(score)
    wr.writerow(line)

    print("Test set metrics:")
    for k in eval_metrics.keys():
        print(k)
        for kk, val in eval_metrics[k].items():
            print(f"{kk}: {val}")
        print('\n')
    torch.save(model.state_dict(), f'./runs/BERT-task-{config.challenge}/best_model.pt')
    print(f"Note classification task ended in: {round(time.process_time() - start, 2)}s")
    metrics_file.close()
