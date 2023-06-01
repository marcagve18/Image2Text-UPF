import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, num_classes=10):
        
        super(MyModel, self).__init__()

        ## Inception
        self.conv11 = nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)
        self.conv12 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3),
            nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, groups=3),
            nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=7, stride=1, padding=3, groups=3),
            nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)
        )
                
        ## VGG 
        self.conv31 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, groups=64),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        )
        self.conv32 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3, groups=128),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        )


        self.conv41 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2, groups=128),
            nn.Conv2d(128, 196, kernel_size=1, stride=1, padding=0)
        )
        self.conv42 = nn.Sequential(
            nn.Conv2d(196, 196, kernel_size=5, stride=1, padding=2, groups=196),
            nn.Conv2d(196, 196, kernel_size=1, stride=1, padding=0)
        )
        
        self.dropout2d = nn.Dropout2d(p=0.05)
        self.dropout = nn.Dropout(p=0.05)
        self.fc = nn.Linear(1764, num_classes, bias=True)
        
        
        self.maxpool= nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        
        ## Inception
        #Inc 1 
        out1 = self.relu(self.conv11(x))
        out2 = self.relu(self.conv12(x))
        out3 = self.relu(self.conv13(x))
        out4 = self.relu(self.conv14(x))
        out = torch.cat([out1, out2, out3, out4],dim=1)
        
        out = self.maxpool(out)
        out = self.dropout2d(out)

        # VGG
        out = self.relu(self.conv31(out))
        out = self.relu(self.conv32(out)+out)
        out = self.maxpool(out)
        out = self.dropout2d(out)

        out = self.relu(self.conv41(out))
        out = self.relu(self.conv42(out)+out)
        out = self.maxpool(out)
        
        #print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.relu(out)
        
        return out


class DecrypterNetwork(nn.Module):
    def __init__( self, hidden_size: int = 8, num_layers=1, num_letters=26, letters_embedding_size: int = 8, use_lstm: bool = False, bidirectional_lstm=False, dropout: float = 0.01):
        # Define RNN or LSTM architecture
        super().__init__()
        self.hidden_size = hidden_size
        self.num_letters = num_letters
        self.letters_embedder = torch.nn.Embedding(num_letters, letters_embedding_size)
        self.use_lstm = use_lstm
        self.bidirectional = bidirectional_lstm
        self.softmax = nn.Softmax(dim=1)
        if use_lstm:
            self.rnn = nn.LSTM( input_size=letters_embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional_lstm, dropout=dropout)
        else:
            self.rnn = nn.RNN( input_size=letters_embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)

        self.last_linear = nn.Linear(hidden_size * 2 if bidirectional_lstm else hidden_size, num_letters)

    def forward(self, X):
        N = X.shape[0]
        L = X.shape[1]

        embedded_letters = self.letters_embedder(X)
        
        # Get hidden states for all letters in the sequence
        hidden_states, _ = self.rnn(embedded_letters)
        
        # In case of multiple input sequneces flat (N,L,hidden_size) to (N*L,hidden_size) for linear layer
        hidden_states_concat = hidden_states.reshape(-1, self.hidden_size * 2 if self.bidirectional else self.hidden_size)
        
        # Get letters probability using the hidden states for each position in the sequence
        letters_loggits = self.last_linear(hidden_states_concat)
        
        # Use soft-max over logits and reshape to format (N,L,num_letteres)
        letters_probs = self.softmax(letters_loggits).reshape(N, L, self.num_letters)
        
        return letters_probs

cnnModel = MyModel(num_classes=128)
lstmModel = DecrypterNetwork()

