import torch
from tqdm.auto import tqdm
import metrics

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
PROGRESS_BAR = tqdm()


def test(test_dataloader, model, vocab_size):
    PROGRESS_BAR.total = len(test_dataloader)

    model.eval()

    loss = 0
    eval_metrics = metrics.LmMetrics(sample_size=len(test_dataloader.sampler))
    for batch in test_dataloader:
        # New labels for linear model PPL estimate (only based on last masked token)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        lm_labels = torch.tensor(create_lm_labels(batch['labels'])).to(DEVICE)

        with torch.no_grad():
            outputs = model(**batch)

        eval_metrics.compute_batch_metrics(mlm_logits=outputs.prediction_logits,
                                           nsp_logits=outputs.seq_relationship_logits,
                                           mlm_labels=batch['labels'],
                                           lm_labels=lm_labels,
                                           nsp_labels=batch['next_sentence_labels'],
                                           vocab_size=vocab_size,
                                           dev=True)
        PROGRESS_BAR.update(1)

        eval_metrics.add_batch()
        loss += outputs.loss.item()

    return eval_metrics.compute(), loss / len(test_dataloader)


def create_lm_labels(batch):
    batch_size, seq_length = batch.shape
    lm_labels = []
    for idx in range(batch_size):
        lm_labels.append([-100] * seq_length)
        for padix in range(seq_length - 1, -1, -1):
            if batch[idx][padix] != -100:
                lm_labels[-1][padix] = batch[idx][padix].item()
                break
    return lm_labels
