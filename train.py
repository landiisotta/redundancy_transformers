import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from eval import test

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
PROGRESS_BAR = tqdm()


def train(train_dataloader, model, optimizer, scheduler):

    model.train()

    loss_batches = 0
    for batch in train_dataloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        PROGRESS_BAR.update(1)
        loss_batches += loss.item()
    return loss_batches


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
        loss = train(train_dataloader, model, optimizer, scheduler) / len(train_dataloader)
        writer.add_scalar('Loss/train', loss, epoch)
        loss_history.append(loss)
    print(loss_history)

    if dev_dataloader:
        ppl, loss = test(dev_dataloader, model, vocab_size)
        writer.add_scalar('Loss/test', loss)
        writer.add_scalar('PPL/test', ppl)
        print(ppl, loss)

    writer.close()
