from torch.utils.data import DataLoader


def run_pretrained(dataset, batch_size, epochs):
    """

    :param dataset: Dataset object from DataDict with fetures
    :param batch_size: Size of the batch
    :return:
    """
    dataset.set_format(type='torch',
                       columns=['input_ids', 'attention_mask',
                                'token_type_ids', 'next_sentence_label',
                                'labels'])

    dataloader = DataLoader(dataset, batch_size=batch_size)

    for epoch in range(epochs):
        for batch in iter(dataloader):
            output = model(**batch)


def train(dataloader, model):
    model.train()
