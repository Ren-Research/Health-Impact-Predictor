import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
from models.fuel_mix_predictor import TimeSeriesTransformer, TimeSeriesLSTM
from models.dispersion_dnn import HealthImpactMLP
from pipeline_and_loss import TransformerWithMLP, CustomLoss
import argparse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===============================
# Data Preparation
# ===============================
def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset."""
    df = pd.read_csv(file_path)

    # Convert percentages to float
    percentage_cols = [
        "Coal_percentage", "Natural_Gas_percentage",
        "Oil_percentage", "Nuclear_percentage", "Renewable_percentage"
    ]
    for col in percentage_cols:
        df[col] = df[col].str.rstrip('%').astype(float) / 100.0

    # Normalize features
    # Normalize features
    for col in percentage_cols:
        if (df[col] == 0).all():  # Check if all values in the column are 0
            df[col] = 0  # Set the entire column to 0
        else:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    # df[percentage_cols] = (df[percentage_cols] - df[percentage_cols].mean()) / df[percentage_cols].std()

    return df

class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_features, target_features, input_steps, output_steps):
        self.data = data
        self.input_features = input_features
        self.target_features = target_features
        self.input_steps = input_steps
        self.output_steps = output_steps

    def __len__(self):
        return len(self.data) - self.input_steps - self.output_steps + 1

    def __getitem__(self, idx):
        x = self.data.iloc[idx: idx + self.input_steps][self.input_features].values
        y = self.data.iloc[idx + self.input_steps: idx + self.input_steps + self.output_steps][
            self.target_features
        ].values
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ===============================
# Model Initialization
# ===============================
def initialize_model(model_type):
    """Initialize the Transformer and MLP models and combine them."""
    if model_type == "transformer":
        fuel_model = TimeSeriesTransformer(input_dim=5, embed_dim=64, num_heads=4)
    else:
        fuel_model = TimeSeriesLSTM(input_dim=5, embed_dim=64, dropout=0.1)
    mlp = HealthImpactMLP(input_dim=5, hidden_dim=128, output_dim=2)
    combined_model = TransformerWithMLP(fuel_model, mlp)
    return combined_model

# ===============================
# Training Loop
# ===============================
def train_model(model, dataloader, loss_fn, optimizer, num_epochs):
    """Train the combined model."""
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_fuel_mix_loss = 0.0
        total_internal_loss = 0.0
        total_external_loss = 0.0

        for batch in dataloader:
            src, targets = batch
            src = src.to(device)
            targets = targets.to(device)
            fuel_mix_target = targets[:, :, :5]  # Assuming the first 5 columns are fuel mix targets
            health_targets = targets[:, :, 5:]  # Assuming the next 2 columns are health cost targets

            # Forward pass
            fuel_mix_pred, health_pred = model(src, fuel_mix_target)

            # Compute loss
            loss, fuel_mix_loss, internal_loss, external_loss = loss_fn(
                fuel_mix_pred=fuel_mix_pred,
                fuel_mix_target=fuel_mix_target,
                health_pred=health_pred,
                health_internal_target=health_targets[:, :, 0],
                health_external_target=health_targets[:, :, 1]
            )

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # other losses
            total_fuel_mix_loss += fuel_mix_loss.item()
            total_internal_loss += internal_loss.item()
            total_external_loss += external_loss.item()

        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            formatted_output = (
                f"Epoch {epoch + 1}/{num_epochs}\n"
                f"  - Loss: {total_loss / len(dataloader):.4f}\n"
                f"  - Fuel Mix Loss: {total_fuel_mix_loss / len(dataloader):.4f}\n"
                f"  - Internal Health Loss: {total_internal_loss / len(dataloader):.4f}\n"
                f"  - External Health Loss: {total_external_loss / len(dataloader):.4f}"
            )
            print(formatted_output)

# === Evaluation Function ===
def evaluate_model(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    total_fuel_mix_loss = 0.0
    total_internal_loss = 0.0
    total_external_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            fuel_mix_target = y_batch[:, :, :5]  # Assuming the first 5 columns are fuel mix targets
            health_targets = y_batch[:, :, 5:]  # Assuming the next 2 columns are health cost targets
            fuel_mix_pred, health_pred = model(x_batch, fuel_mix_target)
            loss, fuel_mix_loss, internal_loss, external_loss = loss_fn(
                fuel_mix_pred=fuel_mix_pred,
                fuel_mix_target=fuel_mix_target,
                health_pred=health_pred,
                health_internal_target=health_targets[:, :, 0],
                health_external_target=health_targets[:, :, 1]
            )
            total_loss += loss.item()
            # other losses
            total_fuel_mix_loss += fuel_mix_loss.item()
            total_internal_loss += internal_loss.item()
            total_external_loss += external_loss.item()

    # avg_loss = total_loss / len(dataloader)

    print(f"Evaluation Loss: {total_loss / len(dataloader):.4f}\n"
          f"Fuel Mix Loss: {total_fuel_mix_loss / len(dataloader):.4f}\n"
          f"Internal Health Loss: {total_internal_loss / len(dataloader):.4f}\n"
          f"External Health Loss: {total_external_loss / len(dataloader):.4f}"
          )
    # return avg_loss

# ===============================
# Main Execution
# ===============================
def main(args):
    # Configuration
    # file_path = "./TVA_dataset.csv"
    file_path = os.path.join(".", f"{args.state}_dataset.csv")
    fulemix_features = [
        "Coal_percentage", "Natural_Gas_percentage",
        "Oil_percentage", "Nuclear_percentage", "Renewable_percentage"
    ]
    # target_features = ["internal_health_cost", "external_health_cost"]
    # health_features = ["internal_health_cost", "external_health_cost"]
    total_features = ["Coal_percentage", "Natural_Gas_percentage", "Oil_percentage", "Nuclear_percentage", "Renewable_percentage",
                      "internal_health_cost", "external_health_cost"]
    input_steps = args.T
    output_steps = args.T
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    beta = args.beta

    # Load and preprocess data
    data = load_and_preprocess_data(file_path)

    # Create dataset and dataloaders
    dataset = TimeSeriesDataset(data, fulemix_features, total_features, input_steps, output_steps)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    model = initialize_model(args.model).to(device)
    loss_fn = CustomLoss(beta=beta)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # # Train the model
    print("Starting training...")
    train_model(model, train_loader, loss_fn, optimizer, num_epochs)

    # path = "./trained_models/CISO_T72_Beta0.9_Epoch50_LSTM.pth"
    # torch.save(model.state_dict(), path)
    # #
    # # # load model
    # model.load_state_dict(torch.load(path))

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, val_loader, loss_fn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="transformer", choices=["transformer", "lstm"],
                        help="Model type to use: 'transformer' or 'lstm'")
    parser.add_argument("--T", type=int, default=24, choices=[24, 72],
                        help="Prediction Time Step")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="weight of fuel mix predictor")
    parser.add_argument("--state", type=str, default="CISO",
                        help="weight of fuel mix predictor")
    args = parser.parse_args()
    main(args)
