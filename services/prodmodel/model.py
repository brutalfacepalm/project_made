from pickle import load as load_pickle
from torch import load as load_torch
from torch import nn, device, cat, FloatTensor, cuda
from utils import load_files_from_s3

device = device('cuda' if cuda.is_available() else 'cpu')

NUM_LABELS = 39

class TextModel(nn.Module):
    def __init__(self, vectors):
        super(TextModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(FloatTensor(vectors), freeze = True)
        self.rnn = nn.LSTM(input_size=300, hidden_size=300, batch_first = True, bidirectional = True)
        self.fc1 = nn.Linear(600, 256)
        self.fc2 = nn.Linear(256, NUM_LABELS)

    def forward(self, text, seq_len):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.rnn(embedded)
        output = cat([hidden[0,:, :], hidden[1,:,:]], dim=1)
        output = nn.functional.relu(self.fc1(output))
        output = self.fc2(output)
        return output



def get_model(path):
    load_files_from_s3('models/baseline/model', path, '/dataformodel/model')
    load_files_from_s3('models/baseline/vectors.pkl', path, '/dataformodel/vectors.pkl')

    with open(path + '/dataformodel/vectors.pkl', 'rb') as vec_file:
        vectors = load_pickle(vec_file)
    
    model = TextModel(vectors)
    model.load_state_dict(load_torch(path + '/dataformodel/model', map_location=device), strict=False)

    return model


