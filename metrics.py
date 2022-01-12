from torch.nn import CrossEntropyLoss
import numpy as np
import torch

ce = CrossEntropyLoss(reduce='none')


class LmMetrics(object):
    model_metrics = {'mlm_loss': [],
                     'lm_loss': [],
                     'nsp_loss': [],
                     'mlm_accuracy': [],
                     'lm_accuracy': [],
                     'nsp_accuracy': [],
                     'lm_entropy': []}

    def __init__(self, sample_size=None):
        if not sample_size:
            print('Metrics are returned as sum over batches. '
                  'To get the average metrics divide by the number of samples.')
        self.sample_size = sample_size
        self.batch_metric = {}

    def compute_batch_metrics(self,
                              mlm_logits,
                              nsp_logits,
                              mlm_labels,
                              nsp_labels,
                              vocab_size,
                              lm_labels=None,
                              dev=False):
        # Loss mlm, nsp
        batch_ce_mlm = ce(mlm_logits.view(-1, vocab_size),
                          mlm_labels.view(-1)).sum()
        batch_ce_nsp = ce(nsp_logits, nsp_labels.view(-1)).sum()
        # Loss sum
        self.batch_metric = {'mlm_loss': batch_ce_mlm.item(),
                             'nsp_loss': batch_ce_nsp.item()}

        if dev:
            # PPL
            batch_ce_lm = ce(mlm_logits.view(-1, vocab_size),
                             lm_labels.view(-1)).sum()
            self.batch_metric['lm_entropy'] = batch_ce_lm
            self.batch_metric['lm_loss'] = batch_ce_lm.item()
            # Accuracy
            mlm_pred = np.argmax(mlm_logits, axis=-1)
            mlm_ref = mlm_labels
            mlm_accuracy = self.__padded_accuracy(pred=mlm_pred, ref=mlm_ref)

            lm_ref = lm_labels
            lm_accuracy = self.__padded_accuracy(pred=mlm_pred, ref=lm_ref)

            nsp_pred = np.argmax(nsp_logits, axis=-1).view(-1)
            nsp_ref = nsp_labels.view(-1)
            nsp_accuracy = sum(i == j for i, j in zip(nsp_pred, nsp_ref))

            self.batch_metric['mlm_accuracy'] = mlm_accuracy.item()
            self.batch_metric['lm_accuracy'] = lm_accuracy.item()
            self.batch_metric['nsp_accuracy'] = nsp_accuracy.item()

    def add_batch(self):
        # Update model_metrics adding the batch metrics
        # LmMetrics.model_metrics update
        for k, val in self.batch_metric.items():
            LmMetrics.model_metrics.setdefault(k, list()).append(val)

    def compute(self):
        if not self.sample_size:
            return {k: sum(val) for k, val in LmMetrics.model_metrics.items()}

        model_metrics = {}
        for k, val in LmMetrics.model_metrics.items():
            if len(val) > 0:
                if k == 'lm_entropy':
                    model_metrics['ppl'] = torch.exp(
                        torch.stack(LmMetrics.model_metrics['lm_entropy']).sum() / self.sample_size).item()
                else:
                    model_metrics[k] = sum(val) / self.sample_size
        return model_metrics

    @staticmethod
    def __padded_accuracy(pred, ref):
        acc = 0
        for p_vect, r_vect in zip(pred, ref):
            lab_len = len([rf for rf in r_vect if rf != -100])
            acc += sum(i == j for i, j in zip(p_vect, r_vect) if j != -100) / lab_len
        return acc
