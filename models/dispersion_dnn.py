import torch
import torch.nn as nn


class HealthImpactMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super(HealthImpactMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # Input x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.shape

        # Flatten the time dimension for MLP input
        x = x.reshape(batch_size * seq_len, input_dim)

        # Pass through the MLP
        x = self.model(x)

        # Reshape back to (batch_size, seq_len, output_dim)
        return x.view(batch_size, seq_len, -1)


