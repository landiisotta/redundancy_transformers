import torch
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
PROGRESS_BAR = tqdm()


def test(test_dataloader, model, vocab_size):
    PROGRESS_BAR.total = len(test_dataloader)

    model.eval()
    ce = CrossEntropyLoss()

    ppl_ce = []
    loss = 0
    for batch in test_dataloader:
        # New labels for linear model PPL estimate (only based on last masked token)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        lm_labels = create_lm_labels(batch['labels'])

        with torch.no_grad():
            outputs = model(**batch)

        mask_pred_logits = outputs.prediction_logits

        batch_ce = ce(mask_pred_logits.view(-1, vocab_size),
                      torch.tensor(lm_labels).to(DEVICE).view(-1))
        ppl_ce.append(batch_ce)
        PROGRESS_BAR.update(1)
        loss += outputs.loss.item()

    ppl = torch.exp(torch.stack(ppl_ce).sum() / len(ppl_ce))
    return ppl.item(), loss / len(test_dataloader)


def create_lm_labels(batch):
    batch_size, seq_length = batch.shape
    lm_labels = []
    for idx in range(batch_size):
        lm_labels.append([-100] * seq_length)
        for padix in range(seq_length - 1, -1, -1):
            if batch[idx][padix] != -100:
                lm_labels[-1][padix] = batch[idx][padix]
                break
    return lm_labels
