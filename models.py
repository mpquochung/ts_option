# models.py
import torch
import torch.nn as nn

class BaseRNNModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        rnn_type: str = "lstm",
        bidirectional: bool = False,
    ):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
        elif self.rnn_type == "rnn":
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity="tanh",
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        self.output_layer = nn.Linear(
            hidden_size * self.num_directions, 1
        )  # predict scalar (price_t+2)

    def forward(self, x):
        """
        x: (B, L, F)
        """
        out, hidden = self.rnn(x)  # out: (B, L, H*num_directions)
        # lấy hidden của time step cuối
        last_out = out[:, -1, :]  # (B, H*num_directions)
        y_hat = self.output_layer(last_out)  # (B, 1)
        return y_hat
