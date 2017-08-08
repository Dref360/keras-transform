class SequentialTransformer():
    def __init__(self, transformers):
        """
        Combine multiple transformers.
        :param transformers: List of SequenceTransformers
        """
        self.transformers = transformers

    def __call__(self, seq):
        for transformer in self.transformers:
            seq = transformer(seq)
        return seq
