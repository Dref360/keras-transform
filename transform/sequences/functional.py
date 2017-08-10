class SequentialTransformer():
    def __init__(self, transformers):
        """
        Combine multiple transformers.
        :param transformers: List of SequenceTransformers
        """
        self.transformers = transformers

    def __call__(self, seq, mask=(True, False)):
        """
        Create a transformer that combines multiples transformers.
        :param seq: Sequence object
        :param mask: Boolean tree-like structure.
        :return: Sequence
        """
        for transformer in self.transformers:
            seq = transformer(seq, mask)
        return seq
