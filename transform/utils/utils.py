def get_batch_size(batch):
    if isinstance(batch, (list, tuple)):
        return get_batch_size(batch[0])
    return batch.shape[0]
