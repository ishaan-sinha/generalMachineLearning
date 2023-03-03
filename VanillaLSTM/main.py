from numpy import array
from torch import optim
from torch.nn import Sequential

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        if hidden == None:
            self.hidden = (torch.zeros(1, 1, self.hidden_size),
                           torch.zeros(1, 1, self.hidden_size))
        else:
            self.hidden = hidden

        lstm_out, self.hidden = self.lstm(x.view(len(x), 1, -1),
                                          self.hidden)

        predictions = self.linear(lstm_out.view(len(x), -1))

        return predictions[-1], self.hidden

def split_sequence(sequence, n_steps):
 X, y = list(), list()
 for i in range(len(sequence)):
     # find the end of this pattern
     end_ix = i + n_steps
     # check if we are beyond the sequence
     if end_ix > len(sequence)-1:
         break
     seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
     X.append(seq_x)
     y.append(seq_y)
 return torch.FloatTensor(X), torch.FloatTensor(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(raw_seq, n_steps)


model = LSTM(input_size=1, hidden_size=50, output_size=1)
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 10
model.train()

for epoch in range(epochs + 1):
    for x, y in zip(X, y):
        x = x.to(device)
        y = y.to(device)
        y_hat, _ = model(x, None)
        optimizer.zero_grad()
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        print(epoch,loss)