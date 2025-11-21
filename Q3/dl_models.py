import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    """
    Input shape: (Batch, Seq_Len, Input_Dim)
    """
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        # We only care about the output of the last time step
        last_time_step = out[:, -1, :]
        prediction = self.fc(last_time_step)
        return prediction

class SimpleCNN(nn.Module):
    """
    1D CNN for Time Series Forecasting.
    Input shape for PyTorch Conv1d: (Batch, Input_Dim, Seq_Len)
    """
    def __init__(self, input_dim=1, num_filters=32, kernel_size=3, output_dim=1, seq_len=30):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
        conv_out_size = seq_len - kernel_size + 1
        self.fc = nn.Linear(num_filters * conv_out_size, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x