import torch
from tqdm.auto import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from transformers import BertForSequenceClassification
from transformers.optimization import get_polynomial_decay_schedule_with_warmup, AdamW, get_scheduler
import sys
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import pickle as pkl
import metrics
import time
import csv
import os

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

        loss, logits = outputs.loss, outputs.logits
        loss.sum().backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        loss_batches += loss.sum().item() * batch['input_ids'].shape[0]

    # Validation
    model.eval()
    loss_batches_eval = 0
    val_metrics = metrics.TaskMetrics(challenge=config.challenge)
    valpred_dict = {}
    vallabel_dict = {}
    for batch in dev_set:
        valnote_ids = batch['note_ids']
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        val_loss, val_output = outputs.loss, outputs.logits
        loss_batches_eval += val_loss.sum().item() * batch['input_ids'].shape[0]
        for i, nid in enumerate(valnote_ids):
            valpred_dict.setdefault(nid, list()).append(val_output[i].tolist())
            if nid not in vallabel_dict:
                vallabel_dict[nid] = batch['labels'][i][0].item()
    if weighting:
        pred, true = _compute_wpred(valpred_dict, vallabel_dict)
    else:

            # Test
            line = fixed_info + ['', '', '']
            model.eval()
            test_metrics = metrics.TaskMetrics(challenge=config.challenge)
            test_loss = 0
            pred_dict = {}
            label_dict = {}
            for test_batch in test_loader:
                note_ids = test_batch['note_ids']
                test_batch = {k: v.to(DEVICE) for k, v in test_batch.items() if k != 'note_ids'}
                with torch.no_grad():
                    outputs = model(**test_batch)
                loss, output = outputs.loss, outputs.logits
                test_loss += loss.sum().item() * test_batch['input_ids'].shape[0]

                # Aggregate outputs by note_id
                for i, nid in enumerate(note_ids):
                    pred_dict.setdefault(nid, list()).append(output[i].tolist())
                    if nid not in label_dict:
                        label_dict[nid] = test_batch['labels'][i][0].item()

            if config.weighting:
                pred, true = _compute_wpred(pred_dict, label_dict)

            else:
                true = []
                pred = torch.tensor([])
                for nid, val in pred_dict.items():
                    pred = torch.cat(pred, torch.mean(torch.tensor(val), dim=0))
                    true.append(label_dict[nid])
            test_metrics.add_batch(true, torch.argmax(pred, dim=-1).tolist())
            eval_metrics = test_metrics.compute()
            writer_test.add_scalar('Test loss', test_loss / (len(test_loader) * GPUS))

    return loss_batches / (len(train_set.sampler) * GPUS), loss_batches_eval / (
            len(dev_set.sampler) * GPUS), train_metrics.compute()


def _normal_density(x, mu, sigma):
    """
    Compute normal density function from z-scores
    """
    return (2. * np.pi * sigma ** 2.) ** -.5 * np.exp(-.5 * (x - mu) ** 2. / sigma ** 2.)


# def _collate_fn(batch):
#     """
#     Custom collate function for DataLoader.
#     """
#     input_ids = batch['input_ids']
#     labels = batch['labels']
#     # Added multi-label case
#     if isinstance(labels[0], list):
#         return {'input_ids': torch.tensor(input_ids),
#                 'labels': torch.tensor(labels)}
#     else:
#         return {'input_ids': torch.tensor(input_ids),
#                 'labels': torch.tensor([[lab] for lab in labels])}


def _collate_fn_batch(batch):
    """
    Custom collate function for DataLoader.
    """
    input_ids, labels, note_ids = [], [], []
    for b in batch:
        input_ids.append(b['input_ids'][0])
        labels.append(b['labels'][0])
        note_ids.append(b['id'][0])
    # Added multi-label case
    if isinstance(labels[0], list):
        return {'input_ids': torch.tensor(input_ids),
                'note_ids': note_ids,
                'labels': torch.tensor([lab for lab in labels])}
    else:
        return {'input_ids': torch.tensor(input_ids),
                'note_ids': note_ids,
                'labels': torch.tensor([[lab] for lab in labels])}


def _compute_wpred(logits_dict, true_labels):
    """
    If weighting method is enabled: compute loss of overlapping note chunks weighting more the initial vectors;
     Otherwise: return batched loss.
    """
    pred = torch.tensor([])
    lab = []
    for nid, val in logits_dict.items():
        # weight initial vectors more
        mid = float(int(len(val) / 2))
        if mid > 0:
            norm_dist = np.array(
                [(i / mid) for i in range(len(val))])
            weights = torch.tensor(np.array([_normal_density(norm_dist, 0, 1)]), device=DEVICE).double()
        else:
            weights = torch.tensor(np.array([[1]]), device=DEVICE).double()
        pred = torch.cat((pred, torch.matmul(weights, torch.tensor(val))))
        lab.append(true_labels[nid])
    return pred, lab


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
    parser.add_argument('--ws_redundancy_train',
                        type=str,
                        help='Word percentage and number of sentences in the training set',
                        dest='ws_redundancy_train')
    parser.add_argument('--ws_redundancy_train',
                        type=str,
                        help='Word percentage and number of sentences in the test set',
                        dest='ws_redundancy_test')
    parser.add_argument('--window_size',
                        type=str,
                        help='Size of the overlapping window',
                        dest='window_size')
    parser.add_argument('--max_seq_length',
                        type=str,
                        help='Maximum sequence length',
                        dest='max_seq_length')
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
    # Create files with tensorboard performances
    tensorboard_folder = f'{config.challenge}{config.ws_redundancy_training}tr{config.ws_redundancy_test}ts'
    writer_train = SummaryWriter(
        f'./runs/BERT-task-{config.challenge}/tensorboard/{tensorboard_folder}/train/{config.learning_rate}')
    writer_val = SummaryWriter(
        f'./runs/BERT-task-{config.challenge}/tensorboard/{tensorboard_folder}/validation/{config.learning_rate}')
    writer_test = SummaryWriter(
        f'./runs/BERT-task-{config.challenge}/tensorboard/{tensorboard_folder}/test/{config.learning_rate}')

    # Load model (5-class classification for smoking challenge; 13-class multi-label for cohort selection)
    if config.challenge == 'smoking_challenge':
        model = BertForSequenceClassification.from_pretrained(config.checkpoint,
                                                              num_labels=config.n_classes)
        # model = model.double()
    elif config.challenge == 'cohort_selection_challenge':
        model = BertForSequenceClassification.from_pretrained(checkpoint=config.checkpoint,
                                                              labels=config.n_classes,
                                                              problem_type="multi_label_classification")
    else:
        print("Challenge not yet implemented... come back later")
        sys.exit()
    # For cohort selection, if class label = MET then that is coded as 1, otherwise NOTMET=1
    if config.challenge == 'cohort_selection_challenge':
        class_label = config.dataset.split('_')[-2]
    else:
        class_label = None
    # Load data pickle
    data = pkl.load(open(config.dataset, 'rb'))
    train, val, test = data['train'], data['validation'], data['test']
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

    # Create documents with results
    colnames = ['ws_training', 'ws_test',
                'learning_rate', 'val_split', 'window_size',
                'sequence_length', 'batch_size', 'weighting_enabled',
                'epoch', 'tr_loss', 'val_loss'] + ['f1_micro', 'f1_macro'] + \
               [f'f1_class{cl}' for cl in range(config.n_classes)] + ['p_micro', 'p_macro'] + \
               [f'p_class{cl}' for cl in range(config.n_classes)] + ['r_micro', 'r_macro'] + \
               [f'r_class{cl}' for cl in range(config.n_classes)]
    if config.challenge == 'cohort_selection_challenge':
        if os.path.exists(f'experiments_{config.challenge}_{class_label}.csv'):
            metrics_file = open(f'experiments_{config.challenge}_{class_label}.csv', 'a')
            wr = csv.writer(metrics_file)
        else:
            metrics_file = open(f'experiments_{config.challenge}_{class_label}.csv', 'a')
            wr = csv.writer(metrics_file)
            wr.writerow(colnames)

    else:
        if os.path.exists(f'experiments_{config.challenge}.csv'):
            metrics_file = open(f'experiments_{config.challenge}.csv', 'a')
            wr = csv.writer(metrics_file)
        else:
            metrics_file = open(f'experiments_{config.challenge}.csv', 'a')
            wr = csv.writer(metrics_file)
            wr.writerow(colnames)

    fixed_info = [config.ws_redundancy_traiing, config.ws_redundancy_test,
                  config.learning_rate, val_split, config.max_sequence_length,
                  config.batch_size, config.weighting]

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
    pred_dict = {}
    label_dict = {}
    for test_batch in test_loader:
        note_ids = test_batch['note_ids']
        test_batch = {k: v.to(DEVICE) for k, v in test_batch.items() if k != 'note_ids'}
        with torch.no_grad():
            outputs = model(**test_batch)
        loss, output = outputs.loss, outputs.logits
        test_loss += loss.sum().item() * test_batch['input_ids'].shape[0]

        # Aggregate outputs by note_id
        for i, nid in enumerate(note_ids):
            pred_dict.setdefault(nid, list()).append(output[i].tolist())
            if nid not in label_dict:
                label_dict[nid] = test_batch['labels'][i][0].item()

    if config.weighting:
        pred, true = _compute_wpred(pred_dict, label_dict)

    else:
        true = []
        pred = torch.tensor([])
        for nid, val in pred_dict.items():
            pred = torch.cat(pred, torch.mean(torch.tensor(val), dim=0))
            true.append(label_dict[nid])
    test_metrics.add_batch(true, torch.argmax(pred, dim=-1).tolist())
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
