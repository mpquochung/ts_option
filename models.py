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
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
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

    def forward(self, x, target_len: int = 1):
        """
        x: (B, L, F)
        """
        out, hidden = self.rnn(x)  # out: (B, L, H*num_directions)
        # lấy hidden của time step cuối
        last_out = out[:, -1, :]  # (B, H*num_directions)
        y_hat = self.output_layer(last_out)  # (B, 1)
        return y_hat


class Seq2SeqModel(nn.Module):
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
            RNN = nn.LSTM
        elif self.rnn_type == "gru":
            RNN = nn.GRU
        elif self.rnn_type == "rnn":
            RNN = nn.RNN
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        self.encoder = RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.decoder = RNN(
            input_size=1,  # previous output
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.output_layer = nn.Linear(
            hidden_size * self.num_directions, 1
        )  # predict scalar

    def forward(self, x, target_len: int):
        """
        x: (B, L, F)
        target_len: int, độ dài sequence cần dự đoán
        """
        batch_size = x.size(0)
        
        # Encoder
        if self.rnn_type == "lstm":
            _, (hidden, cell) = self.encoder(x)
        else:
            # GRU and RNN only return hidden state
            _, hidden = self.encoder(x)
            cell = None

        # Chuẩn bị input ban đầu cho decoder (ví dụ: zeros)
        decoder_input = torch.zeros(batch_size, 1, 1).to(x.device)  # (B, 1, 1)
        outputs = []

        for t in range(target_len):
            if self.rnn_type == "lstm":
                out, (hidden, cell) = self.decoder(
                    decoder_input, (hidden, cell)
                )  # out: (B, 1, H*num_directions)
            else:
                # GRU and RNN
                out, hidden = self.decoder(
                    decoder_input, hidden
                )  # out: (B, 1, H*num_directions)
            y_hat = self.output_layer(out[:, -1, :])  # (B, 1)
            outputs.append(y_hat.unsqueeze(1))  # lưu lại output
            decoder_input = y_hat.unsqueeze(1)  # dùng output làm input cho bước tiếp

        return torch.cat(outputs, dim=1)  # (B, target_len, 1)