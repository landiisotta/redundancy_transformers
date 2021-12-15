from utils import optimizer, lr_scheduler
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train(train_dataloader, model, progress_bar):
    # model.to(device)

    model.train()

    cost = 0
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        cost += loss.item()
    return cost


def train_and_eval(train_dataloader, dev_dataloader, model, n_epochs):
    num_steps = n_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_steps))

    model.to(device)
    loss_history = []
    for epoch in range(n_epochs):
        writer = SummaryWriter()
        loss = train(train_dataloader, model, n_epochs, writer, progress_bar)
        writer.add_scalar('Loss/train', loss, epoch)
        loss_history.append(loss)

        ppl, loss = eval(dev_dataloader, model, )

    writer.flush()
    writer.close()
