import torch
import torch.nn as nn

class GRUModel(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.gru = nn.GRU(
                        self.input_size,
                        self.hidden_size,
                        self.num_layers,
                        bias = True,
                        batch_first=True
                        )

        self.fc1 = nn.Linear(in_features=self.hidden_size, out_features=self.num_classes)

    def forward(self, x:torch.Tensor):
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size)
        # unpacked_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # Pass the packed input through the GRU
        out, _ = self.gru(x, h0)

        out = self.fc1(out)

        return out
    
