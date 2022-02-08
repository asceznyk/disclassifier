import torch
import torch.nn as nn

class BertClassifier(nn.Module):
    def __init__(self, bert, n_class, hidden_dim, dropout=0.1):
        super(BertClassifier, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)  
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(bert.embeddings.word_embeddings.weight.size()[1], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_class)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, seq, mask):
        out = self.bert(seq, attention_mask=mask) 
        x = self.fc1(out['pooler_output']) 
        x = self.relu(x)
        x = self.fc2(x)
        
        return self.log_softmax(x)

class BiGRUClassifier(nn.Module):
    def __init__(self, n_class, vocab_size, hidden_dim, dropout=0.1):
        super(BiGRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 768)
        self.gru = nn.GRU(input_size=768, 
                          hidden_size=hidden_dim, 
                          bidirectional=True)
        self.dropout = nn.Dropout(dropout)  
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_class)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, seq):
        emb = self.embedding(seq)
        out, _ = self.gru(emb.transpose(0,1)) ## time, batch, dim
        x = self.fc1(out[-1, :, :]) ## taking the last time-step as input
        x = self.relu(x)
        x = self.fc2(x)
        return self.log_softmax(x)


