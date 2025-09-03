class DataLoader:

    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        data = open(self.filepath, 'r').read()
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        print("data has %d characters, %d unique" % (data_size, vocab_size))
        char_to_ix = {ch:i for i, ch in enumerate(chars)}
        ix_to_char = {i:ch for i, ch in enumerate(chars)}

        return {
            "data": data,
            "data_size": data_size, 
            "vocab_size": vocab_size, 
            "char_to_ix": char_to_ix, 
            "ix_to_char": ix_to_char
        }