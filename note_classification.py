import torch
from tqdm.auto import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from transformers import BertForSequenceClassification
from transformers.optimization import AdamW, get_scheduler
import sys
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import pickle as pkl
import metrics
import time
import csv
import os
from eval import test_task

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
             challenge,
             method,
             threshold):
    """
    Training/validation step.

    :param train_set:
    :param dev_set:
    :param model:
    :param optimizer:
    :param lr_scheduler:
    :param challenge:
    :return: training and validation loss
    """
    # Training
    model.train()
    loss_batches = 0
    # ctrl = 0
    for batch in train_set:
        batch = {k: v.to(DEVICE) for k, v in batch.items() if k != 'note_ids'}
        outputs = model(**batch)

        loss, logits = outputs.loss, outputs.logits
        loss.sum().backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        loss_batches += loss.sum().item() * batch['input_ids'].shape[0]
        # ctrl += 1
        # if ctrl == 3:
        #     break
    # Validation
    val_metrics = metrics.TaskMetrics(challenge=challenge)
    true_labels, pred_logits, loss_batches_eval = test_task(dev_set, model, challenge)
    lab, pred = _get_labels(true_labels, pred_logits, challenge, method, float(threshold))
    val_metrics.add_batch(lab, pred.tolist())

    return loss_batches / (len(train_set.sampler) * GPUS), loss_batches_eval, val_metrics.compute()


def _normal_density(x, mu, sigma):
    """
    Compute normal density function from z-scores
    """
    return (2. * np.pi * sigma ** 2.) ** -.5 * np.exp(-.5 * (x - mu) ** 2. / sigma ** 2.)


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
                'labels': torch.tensor([lab for lab in labels]).double()}
    else:
        return {'input_ids': torch.tensor(input_ids),
                'note_ids': note_ids,
                'labels': torch.tensor([[lab] for lab in labels])}


def _compute_wpred(true_labels, logits_dict):
    """
    If weighting method is enabled: compute loss of overlapping note chunks weighting more the initial vectors;
     Otherwise: return batched loss.

     :return: list of labels or multi-label lists, tensor of weighted logits (1 x classes dim for each note)
    """
    pred = torch.tensor([]).to(DEVICE)
    lab = []
    for nid, val in logits_dict.items():
        # weight initial vectors more
        mid = float(int(len(val) / 2))
        if mid > 0:
            norm_dist = np.array(
                [(i / mid) for i in range(len(val))])
            weights = torch.tensor(np.array([_normal_density(norm_dist, 0, 1)]), device=DEVICE).float()
        else:
            weights = torch.tensor(np.array([[1]]), device=DEVICE).float()
        pred = torch.cat((pred, torch.matmul(weights, torch.tensor(val).to(DEVICE))))
        lab.append(true_labels[nid])
    return lab, pred


def _get_labels(true_labels, pred_logits, challenge, method=None, threshold=0.5):
    pred = torch.tensor([])
    lab = []
    if method == 'weighting':
        lab, pred = _compute_wpred(true_labels, pred_logits)
    else:
        for nid, val in pred_logits.items():
            if len(val) > 1:
                if method == 'usemax':
                    pred = torch.cat((pred,
                                      torch.max(torch.sigmoid(torch.tensor(val)),
                                                dim=0).values.view(1, -1)))
                else:
                    pred = torch.cat((pred, torch.mean(torch.tensor(val), dim=0).view(1, -1)))
            else:
                pred = torch.cat((pred, torch.tensor(val)))
            lab.append(true_labels[nid])
    if challenge == 'smoking_challenge':
        return lab, torch.argmax(pred, dim=-1)
    elif challenge == 'cohort_selection_challenge':
        return lab, torch.tensor(np.array(pred.cpu() > threshold, dtype=int))


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
    parser.add_argument('--ws_redundancy_test',
                        type=str,
                        help='Word percentage and number of sentences in the test set',
                        dest='ws_redundancy_test')
    parser.add_argument('--window_size',
                        type=int,
                        help='Size of the overlapping window',
                        dest='window_size')
    parser.add_argument('--max_sequence_length',
                        type=str,
                        help='Maximum sequence length',
                        dest='max_sequence_length')
    parser.add_argument('--method',
                        help='Either weighting strategy for smoking challenge or usemax for cohort selection',
                        dest='method',
                        default=None)
    parser.add_argument('--threshold',
                        help='Threshold for multi-label classification',
                        dest='threshold',
                        default=0.5)

    start = time.process_time()

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    config = parser.parse_args(sys.argv[1:])

    # Set class label MET/NOTMET if available
    if config.challenge == 'cohort_selection_challenge':
        class_label = config.dataset.split('_')[-2]
    else:
        class_label = ''
    # Create documents with results
    # Column names
    colnames = ['fold', 'cl_method', 'ws_training', 'ws_test',
                'learning_rate', 'val_split', 'window_size',
                'sequence_length', 'batch_size',
                'epoch', 'tr_loss', 'val_loss'] + ['f1_micro', 'f1_macro'] + \
               [f'f1_class{cl}' for cl in range(config.n_classes)] + ['p_micro', 'p_macro'] + \
               [f'p_class{cl}' for cl in range(config.n_classes)] + ['r_micro', 'r_macro'] + \
               [f'r_class{cl}' for cl in range(config.n_classes)]
    # Files
    if config.challenge == 'cohort_selection_challenge':
        if os.path.exists(f'experiments_{config.challenge}_{class_label}.csv'):
            metrics_file = open(f'experiments_{config.challenge}_{class_label}.csv', 'a')
            wr = csv.writer(metrics_file)
        else:
            metrics_file = open(f'experiments_{config.challenge}_{class_label}.csv', 'w')
            wr = csv.writer(metrics_file)
            wr.writerow(colnames)
    else:
        if os.path.exists(f'experiments_{config.challenge}.csv'):
            metrics_file = open(f'experiments_{config.challenge}.csv', 'a')
            wr = csv.writer(metrics_file)
        else:
            metrics_file = open(f'experiments_{config.challenge}.csv', 'w')
            wr = csv.writer(metrics_file)
            wr.writerow(colnames)

    val_split = ''
    fixed_info = [config.method, config.ws_redundancy_train, config.ws_redundancy_test,
                  config.learning_rate, val_split, str(config.window_size), config.max_sequence_length,
                  config.batch_size]

    # If only evaluate
    if config.ws_redundancy_train != config.ws_redundancy_test:
        best_model_dir = f'./runs/BERT-task-{config.challenge}/redu' \
                         f'{config.ws_redundancy_train}tr{config.ws_redundancy_train}ts{class_label}_' \
                         f'maxseqlen{config.max_sequence_length}_' \
                         f'{config.learning_rate}'

        if config.challenge == 'smoking_challenge':
            data_path = f'./datasets/2006_smoking_status/2006_smoking_status_task_dataset_maxlen' \
                        f'{config.max_sequence_length}'
            model = BertForSequenceClassification.from_pretrained(best_model_dir,
                                                                  num_labels=config.n_classes,
                                                                  from_tf=False)
        elif config.challenge == 'cohort_selection_challenge':
            data_path = f'./datasets/2018_cohort_selection/2018_cohort_selection_task_dataset_' \
                        f'{class_label}_maxlen{config.max_sequence_length}'
            model = BertForSequenceClassification.from_pretrained(best_model_dir,
                                                                  num_labels=config.n_classes,
                                                                  problem_type="multi_label_classification",
                                                                  from_tf=False)
        else:
            data_path = ''
        data = pkl.load(open(data_path + f'{config.ws_redundancy_test}windowsize{str(config.window_size)}.'
                                         f'pkl', 'rb'))
        model.to(DEVICE)
        testset = data['test']
        test_loader = DataLoader(testset,
                                 shuffle=False,
                                 collate_fn=_collate_fn_batch,
                                 batch_size=config.batch_size)
        # Test
        line = ['test'] + fixed_info + ['', '', '']
        test_metrics = metrics.TaskMetrics(challenge=config.challenge)
        true_labels, pred_logits, test_loss = test_task(test_loader, model, config.challenge)
        print(f"Model trained on redundancy {config.ws_redundancy_train}, tested on redundancy "
              f"{config.ws_redundancy_test}:")
        print(f"Test set loss: {test_loss}")
        true, pred = _get_labels(true_labels, pred_logits, config.challenge, config.method, float(config.threshold))
        test_metrics.add_batch(true, pred.tolist())
        eval_metrics = test_metrics.compute()

        for k in eval_metrics.keys():
            for kk, score in eval_metrics[k].items():
                line.append(score)
        wr.writerow(line)

        print("Test set metrics:")
        for k in eval_metrics.keys():
            print(k)
            for kk, val in eval_metrics[k].items():
                print(f"{kk}: {val}")
            print('\n')

        print(f"Model evaluation task ended in: {round(time.process_time() - start, 2)}s")
        metrics_file.close()
        sys.exit()

    # Create files with tensorboard performances
    # For cohort selection, if class label = MET then that is coded as 1, otherwise NOTMET=1
    tensorboard_folder = f'redu{config.ws_redundancy_train}tr{config.ws_redundancy_test}ts{class_label}'
    writer_train = SummaryWriter(
        f'./runs/BERT-task-{config.challenge}/tensorboard{config.max_sequence_length}/'
        f'train/{tensorboard_folder}_{config.learning_rate}')
    writer_val = SummaryWriter(
        f'./runs/BERT-task-{config.challenge}/tensorboard{config.max_sequence_length}/'
        f'validation/{tensorboard_folder}_{config.learning_rate}')

    # Load model (5-class classification for smoking challenge; 13-class multi-label for cohort selection)
    if config.challenge == 'smoking_challenge':
        model = BertForSequenceClassification.from_pretrained(config.checkpoint,
                                                              num_labels=config.n_classes,
                                                              from_tf=False)
    elif config.challenge == 'cohort_selection_challenge':
        model = BertForSequenceClassification.from_pretrained(config.checkpoint,
                                                              num_labels=config.n_classes,
                                                              problem_type="multi_label_classification",
                                                              from_tf=False)
    else:
        print("Challenge not yet implemented... come back later")
        sys.exit()

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
    print(f"Method: {config.method}\n")

    progress_bar = tqdm(range(num_training_steps))

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_scheduler('linear',
                                 num_warmup_steps=warmup_steps,
                                 num_training_steps=num_training_steps,
                                 optimizer=optimizer)

    # Run on multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    val_split = round((val.num_rows * 100) / (val.num_rows + train.num_rows), 1)
    fixed_info = [config.method, config.ws_redundancy_train, config.ws_redundancy_test,
                  config.learning_rate, val_split, str(config.window_size), config.max_sequence_length,
                  config.batch_size]

    model.to(DEVICE)
    # Training
    for epoch in range(config.epochs):
        tr_loss, val_loss, val_metrics = training(train_loader,
                                                  val_loader,
                                                  model,
                                                  optimizer,
                                                  lr_scheduler,
                                                  config.challenge,
                                                  config.method,
                                                  config.threshold)
        line = ['train/val'] + fixed_info + [epoch, tr_loss, val_loss]
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
            best_model_dir = f'./runs/BERT-task-{config.challenge}/' \
                             f'redu{config.ws_redundancy_train}tr' \
                             f'{config.ws_redundancy_test}ts{class_label}_maxseqlen{config.max_sequence_length}_' \
                             f'{config.learning_rate}'
            os.makedirs(best_model_dir, exist_ok=True)
            if epoch != (config.epochs - 1) and epoch != 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': tr_loss}, f'{best_model_dir}/checkpoint.pt')
            else:
                # Comment line and uncomment next if multiple GPUs are used
                if GPUS == 1:
                    model.save_pretrained(best_model_dir)
                elif GPUS > 1:
                    model.module.save_pretrained(best_model_dir)

        wr.writerow(line)
    # Test
    line = ['test'] + fixed_info + ['', '', '']
    test_metrics = metrics.TaskMetrics(challenge=config.challenge)
    true_labels, pred_logits, test_loss = test_task(test_loader, model, config.challenge)
    best_model_dir = f'./runs/BERT-task-{config.challenge}/' \
                     f'redu{config.ws_redundancy_train}tr' \
                     f'{config.ws_redundancy_test}ts{class_label}_maxseqlen{config.max_sequence_length}_' \
                     f'{config.learning_rate}'
    pkl.dump(pred_logits, open(os.path.join(best_model_dir, 'pred_logits.pkl'), 'wb'))
    print(f"Test set loss: {test_loss}")
    true, pred = _get_labels(true_labels, pred_logits, config.challenge, config.method, float(config.threshold))
    test_metrics.add_batch(true, pred.tolist())
    eval_metrics = test_metrics.compute()

    for k in eval_metrics.keys():
        for kk, score in eval_metrics[k].items():
            line.append(score)
    wr.writerow(line)

    print("Test set metrics:")
    for k in eval_metrics.keys():
        print(k)
        for kk, val in eval_metrics[k].items():
            print(f"{kk}: {val}")
        print('\n')

    print(f"Note classification task ended in: {round(time.process_time() - start, 2)}s")
    metrics_file.close()
