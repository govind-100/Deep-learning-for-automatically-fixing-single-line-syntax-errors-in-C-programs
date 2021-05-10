class Vocabulary:
    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token
    OOV_token = 3  # OOV token

    def __init__(self):
        self.word2index = {"PAD": self.PAD_token, "SOS": self.SOS_token, "EOS": self.EOS_token, "OOV": self.OOV_token}
        self.word2count = {"PAD": 0, "SOS": 1, "EOS": 2, "OOV": 3}
        self.index2word = {self.PAD_token: "PAD", self.SOS_token: "SOS", self.EOS_token: "EOS", self.OOV_token: "OOV"}
        self.num_words = 4
        self.num_sentences = 0
        self.longest_sentence = 0

    def add_word(self, word):
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1

    def createVoc(self, tokens):
        sentence_len = 0
        # print(tokens)
        for token in tokens:
            self.add_word(token)

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]
