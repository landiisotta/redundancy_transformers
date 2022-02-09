import torch
from tqdm.auto import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from transformers import AutoModelForSequenceClassification, AdamW, get_scheduler
import sys
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import pickle as pkl
import metrics
import time

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
GPUS = torch.cuda.device_count()


# Smoking challenge
# Measures to evaluate classification: Precision, Recall, F1 score
# Evaluation on each smoking category separately, then microaverage and macroaverage (for all metrics)
def training(train_set,
             dev_set,
             model,
             optimizer,
             lr_scheduler):
    """
    Training/validation step.

    :param train_set:
    :param dev_set:
    :param model:
    :param optimizer:
    :param lr_scheduler:
    :return: training and validation loss
    """
    # Training
    model.train()
    loss_batches = 0
    train_metrics = metrics.TaskMetrics(challenge='smoking_challenge')
    for batch in train_set:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)

        wloss, _ = _compute_wloss(batch, outputs)

        wloss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        loss_batches += wloss.sum().item()

    # Validation
    model.eval()
    loss_batches_eval = 0
    for batch in dev_set:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        val_loss, output_logits = _compute_wloss(batch, outputs)
        loss_batches_eval += val_loss.sum().item()

        train_metrics.add_batch(batch['labels'][0].item(),
                                torch.argmax(output_logits).item())

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
    return {'input_ids': torch.tensor(input_ids),
            'labels': torch.tensor([[lab] for lab in labels])}


def _compute_wloss(batch, outputs):
    """
    Compute loss of overlapping note chunks weighting more the initial vectors.
    """
    if batch['input_ids'].shape[0] > 1:
        # weight initial vectors more
        mid = float(int(batch['input_ids'].shape[0] / 2))
        norm_dist = np.array(
            [(i / mid) for i in range(batch['input_ids'].shape[0])])

        weights = torch.tensor(np.array([_normal_density(norm_dist, 0, 1)]), device=DEVICE).double()
        wloss = nn.functional.cross_entropy(torch.matmul(weights, outputs.logits),
                                            batch['labels'][0].view(-1))
        output_logits = torch.matmul(weights, outputs.logits)
    else:
        wloss = outputs.loss
        output_logits = outputs.logits

    return wloss, output_logits


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
                        help='Number of classes',
                        dest='n_classes')

    start = time.process_time()

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    config = parser.parse_args(sys.argv[1:])
    writer_train = SummaryWriter('./runs/BERT-task-smoking/tensorboard/train')
    writer_val = SummaryWriter('./runs/BERT-task-smoking/tensorboard/validation')

    writer_test = SummaryWriter('./runs/BERT-task-smoking/tensorboard/test')

    model = AutoModelForSequenceClassification.from_pretrained(config.checkpoint,
                                                               num_labels=config.n_classes)
    model = model.double()

    data = pkl.load(open(config.dataset, 'rb'))
    train, val, test = data['train'], data['validation'], data['test']

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
    progress_bar = tqdm(range(num_training_steps))

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                                 num_warmup_steps=round(num_training_steps / 100),
                                 num_training_steps=num_training_steps)

    # Run on multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    model.to(DEVICE)
    # Training
    for epoch in range(config.epochs):
        tr_loss, val_loss, val_metrics = training(train_loader,
                                                  val_loader,
                                                  model,
                                                  optimizer,
                                                  lr_scheduler)
        writer_train.add_scalar('epoch_loss', tr_loss, epoch)
        writer_val.add_scalar('epoch_loss', val_loss, epoch)
        for k in val_metrics.keys():
            for kk, score in val_metrics[k].items():
                writer_val.add_scalar(f'Validation {kk}', score, epoch)
        if epoch % 2 == 0:
            print(f"Epoch {epoch} -- Training loss {round(tr_loss, 4)}")
            print(f"Epoch {epoch} -- Validation loss {round(val_loss, 4)}")
            print('\n')
            print(f"Epoch {epoch} -- Classification metrics:")
            for k in val_metrics.keys():
                print(k)
                for kk, val in val_metrics[k].items():
                    print(f"{kk}: {val}")
                print('\n')
    # Test
    model.eval()
    test_metrics = metrics.TaskMetrics(challenge='smoking_challenge')
    test_loss = 0

    for test_batch in test_loader:
        with torch.no_grad():
            outputs = model(**test_batch)
        wloss, output_logits = _compute_wloss(test_batch, outputs)
        test_metrics.add_batch(test_batch['labels'][0].item(), torch.argmax(output_logits).item())
        test_loss += wloss.sum()

    eval_metrics = test_metrics.compute()

    writer_test.add_scalar('Test loss', test_loss / (len(test_loader.sampler) * GPUS))
    for k in eval_metrics.keys():
        for kk, score in eval_metrics[k].items():
            writer_test.add_scalar(f'Test {kk}', score)

    print("Test set metrics:")
    for k in test_metrics.keys():
        print(k)
        for kk, val in test_metrics[k].items():
            print(f"{kk}: {val}")
        print('\n')

    print(f"Note classification task ended in: {round(time.process_time() - start, 2)}s")
