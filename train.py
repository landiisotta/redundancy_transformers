import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from eval import test
import metrics

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
GPUS = max(torch.cuda.device_count(), 1)
PROGRESS_BAR = tqdm()


def train(train_dataloader, vocab_size, model, optimizer, scheduler):
    """Training pass"""
    model.train()

    loss_batches = 0
    train_metrics = metrics.LmMetrics(sample_size=len(train_dataloader.sampler))
    for batch in train_dataloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.sum().backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        PROGRESS_BAR.update(1)

        loss_batches += loss.sum().item() * batch['input_ids'].shape[0]
        train_metrics.compute_batch_metrics(mlm_logits=outputs.prediction_logits.detach(),
                                            nsp_logits=outputs.seq_relationship_logits.detach(),
                                            mlm_labels=batch['labels'].detach(),
                                            nsp_labels=batch['next_sentence_label'].detach(),
                                            vocab_size=vocab_size,
                                            dev=False)
        train_metrics.add_batch()
    return train_metrics.compute(), loss_batches / (len(train_dataloader.sampler) * GPUS)


def train_and_eval(train_dataloader,
                   dev_dataloader,
                   model,
                   vocab_size,
                   optimizer,
                   scheduler,
                   patience,
                   n_epochs=10):
    """Run training and evaluate on dev dataset"""
    num_steps = n_epochs * len(train_dataloader)
    PROGRESS_BAR.total = num_steps
    # Save performance in Tensorboard
    writer_train = SummaryWriter('./runs/BERT-fine-tuning/tensorboard/train')
    writer_val = SummaryWriter('./runs/BERT-fine-tuning/tensorboard/validation')

    loss_history = []
    best_dev_loss = 1e15
    c_patience = 0
    stop_training = False
    stopped_epoch = 0
    epoch_chkpt = 0
    for epoch in range(n_epochs):
        # Train
        train_metrics, loss = train(train_dataloader, vocab_size, model, optimizer, scheduler)
        writer_train.add_scalar('epoch_loss', loss, epoch)
        for k, val in train_metrics.items():
            writer_train.add_scalar(f'{k}', val, epoch)
        loss_history.append(loss)
        if epoch % 10 == 0:
            print("\n")
            print(f"Epoch: {epoch} -- Train metrics: {train_metrics}")
            print(f"Epoch {epoch} -- Train loss: {loss}")
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, f'./runs/BERT-fine-tuning/checkpoint_resume_epoch{epoch}.pt')
        # Validate with early stopping if overfitting
        if dev_dataloader:
            eval_metrics, loss = test(dev_dataloader, model, vocab_size)
            current_loss = loss
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} -- Validation metrics: {eval_metrics}")
                print(f"Epoch {epoch} -- Validation loss: {loss}")
            if (current_loss - best_dev_loss) < 0:
                c_patience = 0
                best_dev_loss = current_loss
                epoch_chkpt = epoch
                torch.save(model.state_dict(),
                           './runs/BERT-fine-tuning/checkpoint.pt')
            else:
                if c_patience == patience:
                    writer_val.add_scalar('epoch_loss', loss, epoch)
                    for k, val in eval_metrics.items():
                        writer_val.add_scalar(f'{k}', val, epoch)
                    model.load_state_dict(torch.load('./runs/BERT-fine-tuning/checkpoint.pt'))
                    stopped_epoch = epoch
                    stop_training = True
                    break
                c_patience += 1
            writer_val.add_scalar('epoch_loss', loss, epoch)
            for k, val in eval_metrics.items():
                writer_val.add_scalar(f'{k}', val, epoch)

    if stop_training:
        print("\n")
        print(f"Training stopped at {stopped_epoch} -- Best model at epoch {epoch_chkpt}")
        eval_metrics, loss = test(dev_dataloader, model, vocab_size)
        print(f"Metrics at early stopped epoch {epoch_chkpt}: {eval_metrics}")
        print(f"Loss at early stopped epoch {epoch_chkpt}: {loss}")

    # torch.save(model.state_dict(), './runs/BERT-fine-tuning/best_model.pt')
    model.module.save_pretrained('./runs/BERT-fine-tuning')

    writer_train.close()
    writer_val.close()
