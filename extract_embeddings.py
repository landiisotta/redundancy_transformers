from transformers import BertForPreTraining
import pickle as pkl
from torch.utils.data import DataLoader
import torch
import time
from tqdm import tqdm

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
PROGRESS_BAR = tqdm()

dt, tkn = pkl.load(open('./datasets/n2c2_datasets/n2c2datasets_forClinicalBERTfinetuning_maxseqlen12800.pkl', 'rb'))
with open('./datasets/n2c2_datasets/test_newk_to_oldk.csv', 'r') as f:
    ch_dict = {}
    next(f)
    for line in f:
        idx, _, ch = line.strip('\n').split(',')
        ch_dict[idx] = ch

test = dt['test']
test.set_format(type='torch',
                columns=['input_ids',
                         'attention_mask',
                         'token_type_ids',
                         'next_sentence_label',
                         'labels'])
test_loader = DataLoader(test, shuffle=False, batch_size=8)
len_loader = len(test_loader)

model = BertForPreTraining.from_pretrained('./runs/BERT-fine-tuning/redu00tr00ts', from_tf=False)
model.to(DEVICE)
model.eval()

embeddings_4layers = torch.tensor([]).to(DEVICE)
embeddings_2ndtolast = torch.tensor([]).to(DEVICE)
note_ids = []
# i = 0
checkpoint = time.process_time()
for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
    # i += 1
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    batch_size = len(batch['input_ids'])
    with torch.no_grad():
        out = model(**batch, output_hidden_states=True)
        # Concat last four hidden layers of [CLS] token and average by note
        embeddings_4layers = torch.cat([embeddings_4layers,
                                        torch.concat([out.hidden_states[-j][:, 0] for j in range(-4, 0)], dim=1)],
                                       dim=0)
        # Extract second to last layer
        embeddings_2ndtolast = torch.cat([embeddings_2ndtolast,
                                          out.hidden_states[-2][:, 0]], dim=0)
        note_ids.extend(test['note_id'][idx * batch_size:batch_size * (idx + 1)])
    # if i == 10:
    #     break
# print(f"Time estimated for model evaluation: {round(time.process_time() - checkpoint) * (len_loader/10)}")
print(f"Time for model evaluation: {round(time.process_time() - checkpoint)}")

emb_4layers = {}
emb_2ndtolast = {}
for idx, k in enumerate(note_ids):
    emb_4layers.setdefault(k, list()).append(embeddings_4layers[idx])
    emb_2ndtolast.setdefault(k, list()).append(embeddings_2ndtolast[idx])

f4 = open('embeddings_4layerconcat.csv', 'w')
for k, t in emb_4layers.items():
    if len(t) > 1:
        tm = torch.mean(torch.cat(t).view(len(t), -1), dim=0)
    else:
        tm = t[0]
    # try:
    f4.write(','.join([k, ch_dict[k]] + [str(el.item()) for el in tm]) + '\n')
    # except KeyError:
    #     f4.write(','.join([k, ''] + [str(el.item()) for el in tm]) + '\n')
f4.close()

f2 = open('embeddings_2tolast.csv', 'w')
for k, t in emb_2ndtolast.items():
    if len(t) > 1:
        tm = torch.mean(torch.cat(t).view(len(t), -1), dim=0)
    else:
        tm = t[0]
    # try:
    f2.write(','.join([k, ch_dict[k]] + [str(el.item()) for el in tm]) + '\n')
    # except KeyError:
    #     f2.write(','.join([k, ''] + [str(el.item()) for el in tm]) + '\n')
f2.close()
