import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from eval import test
import metrics

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
PROGRESS_BAR = tqdm()


def train(train_dataloader, vocab_size, model, optimizer, scheduler):
    model.train()

    loss_batches = 0
    train_metrics = metrics.LmMetrics(sample_size=len(train_dataloader.sampler))
    for batch in train_dataloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        PROGRESS_BAR.update(1)

        loss_batches += loss.item() * batch['input_ids'].shape[0]
        train_metrics.compute_batch_metrics(mlm_logits=outputs.prediction_logits.detach(),
                                            nsp_logits=outputs.seq_relationship_logits.detach(),
                                            mlm_labels=batch['labels'].detach(),
                                            nsp_labels=batch['next_sentence_labels'].detach(),
                                            vocab_size=vocab_size,
                                            dev=False)
        train_metrics.add_batch()
    return train_metrics.compute(), loss_batches / len(train_dataloader.sampler)


def train_and_eval(train_dataloader,
                   dev_dataloader,
                   model,
                   vocab_size,
                   optimizer,
                   scheduler,
                   n_epochs=10):
    num_steps = n_epochs * len(train_dataloader)
    PROGRESS_BAR.total = num_steps
    writer = SummaryWriter('./runs/BERT-fine-tuning')

    loss_history = []

    for epoch in range(n_epochs):
        # Train
        train_metrics, loss = train(train_dataloader, vocab_size, model, optimizer, scheduler)
        writer.add_scalar('Loss/train', loss, epoch)
        for k, val in train_metrics:
            writer.add_scalar(f'{k}/train', val, epoch)
        loss_history.append(loss)
        print(loss_history)

        # Validate
        if dev_dataloader:
            eval_metrics, loss = test(dev_dataloader, model, vocab_size)
            writer.add_scalar('Loss/test', loss, epoch)
            for k, val in eval_metrics.items():
                writer.add_scalar(f'{k}/test', val, epoch)
            print(eval_metrics, loss)

    writer.close()
