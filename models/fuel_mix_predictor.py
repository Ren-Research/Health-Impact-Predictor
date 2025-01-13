import torch
import torch.nn as nn

# Define the Transformer model
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim, nhead=num_heads, num_encoder_layers=1, num_decoder_layers=1, dropout=dropout
        )
        self.fc_out = nn.Linear(embed_dim, input_dim)

    def forward(self, src, tgt):
        # Embed the input and target
        src_embed = self.embedding(src)
        tgt_embed = self.embedding(tgt)

        # Transpose to match the transformer's input shape
        src_embed = src_embed.permute(1, 0, 2)  # (seq_len, batch, embed_dim)
        tgt_embed = tgt_embed.permute(1, 0, 2)

        # Pass through the transformer
        transformer_out = self.transformer(src_embed, tgt_embed)

        # Final output layer
        output = self.fc_out(transformer_out)
        return output.permute(1, 0, 2)  # Back to (batch, seq_len, input_dim)


# # Hyperparameters
# input_dim = 5  # Number of features
# embed_dim = 64  # Embedding dimension
# num_heads = 4  # Number of attention heads
# sequence_length = 24
# learning_rate = 0.001
# epochs = 20
#
# # Prepare the data
# filename = "fuel_mix_dataset.csv"  # Replace with your dataset filename
# inputs, targets, scaler = preprocess_data(filename, sequence_length=sequence_length)
#
# # Split into training and validation sets
# train_size = int(0.8 * len(inputs))
# train_inputs, val_inputs = inputs[:train_size], inputs[train_size:]
# train_targets, val_targets = targets[:train_size], targets[train_size:]
#
# # Initialize the model, loss function, and optimizer
# model = TimeSeriesTransformer(input_dim, embed_dim, num_heads)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# # Training loop
# for epoch in range(epochs):
#     model.train()
#     optimizer.zero_grad()
#
#     # Teacher forcing: Provide the input as both src and tgt
#     output = model(train_inputs, train_inputs)
#
#     # Calculate loss
#     loss = criterion(output, train_targets)
#     loss.backward()
#     optimizer.step()
#
#     # Validation
#     model.eval()
#     with torch.no_grad():
#         val_output = model(val_inputs, val_inputs)
#         val_loss = criterion(val_output, val_targets)
#
#     print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
#
# # Predict the next 24 hours
# model.eval()
# with torch.no_grad():
#     predictions = model(val_inputs[-1:], val_inputs[-1:]).numpy()
#     predictions = scaler.inverse_transform(predictions[0])  # Rescale to original percentages
#
# print("Predicted Fuel Mix for Next 24 Hours:", predictions)
