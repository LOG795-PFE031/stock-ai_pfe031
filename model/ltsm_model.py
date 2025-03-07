from torch import nn

class LSTMStocksModule(nn.Module):
    HIDDEN_SIZE = 100  # Match TensorFlow model (100 units)
    NUM_LAYERS = 1
    BIAS = True

    def __init__(self):
        super(LSTMStocksModule, self).__init__()
        self.lstm = nn.LSTM(
            1,  # Keep as 1 since we only use Open price for prediction
            self.HIDDEN_SIZE,
            self.NUM_LAYERS,
            self.BIAS,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.2)  # Match TensorFlow dropout rate
        self.linear = nn.Linear(self.HIDDEN_SIZE, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])  # Use last time step and apply dropout
        out = self.linear(lstm_out)
        return out