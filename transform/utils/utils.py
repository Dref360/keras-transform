def get_batch_size(batch):
    """Get the batch size from a tree structure."""
    if isinstance(batch, (list, tuple)):
        return get_batch_size(batch[0])
    return batch.shape[0]
