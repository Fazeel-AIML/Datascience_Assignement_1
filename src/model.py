import torch
import torch.nn as nn

class BoW(nn.Module):
    def __init__(self, nwords, ntags):
        super(BoW, self).__init__()
        self.embedding = nn.Embedding(nwords, ntags)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.bias = nn.Parameter(torch.zeros(ntags))
    
    def forward(self, x):
        emb = self.embedding(x)
        out = torch.sum(emb, dim=0) + self.bias
        return out.view(1, -1)

def create_tensors(data, word_to_index, tag_to_index):
    for line in data:
        yield ([word_to_index.get(word, word_to_index["<unk>"]) for word in line[0]], tag_to_index[line[1]])

