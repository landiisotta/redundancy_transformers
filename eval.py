import torch
from torch.nn import CrossEntropyLoss


def eval(test_dataloader, model, writer, progress_bar):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    ppl_ce = []
    ce = CrossEntropyLoss()
    loss = []
    for batch in test_dataloader:
        # New labels for linear model PPL estimate (only based on last masked token)
        batch = {k: v.to(device) for k, v in batch.items()}
        lm_labels = create_lm_labels(batch['labels'])

        with torch.no_grad():
            outputs = model(**batch)

        mask_pred_logits = outputs.prediction_logits
        nsp_logits = outputs.seq_relationship_logits

        batch_ce = ce(mask_pred_logits.view(-1, tokenizer.vocab_size),
                         torch.tensor(lm_labels).view(-1))
        ppl_ce.append(batch_ce)
        progress_bar(1)
        loss += outputs.loss.item()
    ppl = torch.exp(torch.stack(ppl_ce).sum() / len(ppl_ce))
    writer.add_scalar('Loss/test', loss)
    writer.add_scalar('PPL/test', ppl)


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
