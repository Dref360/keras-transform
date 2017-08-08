def get_batch_size(batch):
    if isinstance(batch, (list, tuple)):
        return get_batch_size(batch[0])
    return batch.shape[0]


def get_batch_shape(batch):
    if isinstance(batch, (list, tuple)):
        return [get_batch_size(batch[i]) for i in range(len(batch))]
    return batch.shape
