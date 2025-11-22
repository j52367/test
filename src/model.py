import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=512, output_size=21):
        super(BaseModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size//2, output_size)
        )
        self.actv = nn.ReLU()

    def forward(self, x):
        # x shape: (B, TRAIN_WINDOW_SIZE, 5)
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size, x.device)

        # GRU layer
        gru_out, hidden = self.gru(x, hidden)

        # Only use the last output sequence
        last_output = gru_out[:, -1, :]

        # Fully connected layer
        output = self.actv(self.fc(last_output))

        return output.squeeze(1)

    def init_hidden(self, batch_size, device):
        # Initialize hidden state
        return torch.zeros(1, batch_size, self.hidden_size, device=device)