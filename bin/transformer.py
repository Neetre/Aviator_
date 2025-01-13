import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import math
from torch.amp import autocast
from torch.optim.lr_scheduler import OneCycleLR
import json
from tqdm import tqdm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class MultivariateTSDataset(Dataset):
    def __init__(self, data, seq_length):
        """
        data: pandas DataFrame with columns [color, mean, var, next_approximate, target]
        """
        self.seq_length = seq_length

        self.means = {
            'mean': data['mean'].mean(),
            'var': data['var'].mean(),
            'next_approximate': data['next_approximate'].mean(),
            'target': data['target'].mean()
        }
        self.stds = {
            'mean': data['mean'].std(),
            'var': data['var'].std(),
            'next_approximate': data['next_approximate'].std(),
            'target': data['target'].std()
        }
        
        normalized_data = []
        for _, row in data.iterrows():
            feature_vector = [
                row['color'],
                (row['mean'] - self.means['mean']) / self.stds['mean'],
                (row['var'] - self.means['var']) / self.stds['var'],
                (row['next_approximate'] - self.means['next_approximate']) / self.stds['next_approximate']
            ]
            normalized_data.append(feature_vector)
            
        self.data = torch.tensor(normalized_data, dtype=torch.float32)
        self.targets = torch.tensor(
            (data['target'].values - self.means['target']) / self.stds['target'],
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.seq_length]
        target = self.targets[idx + self.seq_length]
        return sequence, target


class MultivariateTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*8,
            dropout=dropout,
            batch_first=True,
            activation=nn.GELU()
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, 1)
        )

    def forward(self, src, src_mask=None):
        x = self.embedding(src)
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_mask)
        x = self.decoder(x[:, -1, :])
        return x


def prepare_data_from_csv(file_path, seq_length, batch_size):
    df = pd.read_csv(file_path)

    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    
    train_data = df.iloc[:train_size]
    val_data = df.iloc[train_size:train_size + val_size]
    test_data = df.iloc[train_size + val_size:]

    train_dataset = MultivariateTSDataset(train_data, seq_length)
    val_dataset = MultivariateTSDataset(val_data, seq_length)
    test_dataset = MultivariateTSDataset(test_data, seq_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, 
                          pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, 
                           pin_memory=True)
    
    return (train_loader, val_loader, test_loader, 
            train_dataset.means, train_dataset.stds)


def predict_next_value(model, sequence_str, means, stds, device):
    """
    sequence_str: comma-separated string with format: "color,mean,var,next_approximate"
    """
    try:
        rows = [row.strip().split(',') for row in sequence_str.split(';')]
        if len(rows) != model.seq_length - 1:
            raise ValueError(f"Please provide exactly {model.seq_length-1} sequences")

        sequence = []
        for row in rows:
            if len(row) != 4:
                raise ValueError("Each row should have 4 values: color,mean,var,next_approximate")
            
            feature_vector = [
                float(row[0]),
                (float(row[1]) - means['mean']) / stds['mean'],
                (float(row[2]) - means['var']) / stds['var'],
                (float(row[3]) - means['next_approximate']) / stds['next_approximate']
            ]
            sequence.append(feature_vector)

        input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            with autocast():
                output = model(input_tensor)

            prediction = output.cpu().item() * stds['target'] + means['target']
            return prediction
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


def interactive_prediction(model, means, stds, device):
    print("\nMultivariate Transformer Prediction Interface")
    print("Please input sequences in the format: color,mean,var,next_approximate")
    print("Separate sequences with semicolons (;)")
    print("Example: 1,1.2,0.3,1.5; 0,1.3,0.4,1.6; ...")
    
    while True:
        try:
            user_input = input("\nEnter your sequences (or 'q' to quit): ")
            
            if user_input.lower() == 'q':
                break
                
            prediction = predict_next_value(model, user_input, means, stds, device)
            
            if prediction is not None:
                print(f"\nPredicted target value: {prediction:.4f}")
                
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again with valid input")


def create_attention_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 1


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, patience=10):
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in tqdm(num_epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.unsqueeze(-1).to(device)
            
            mask = create_attention_mask(inputs.size(1)).to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, mask)
            loss = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.unsqueeze(-1).to(device)
                mask = create_attention_mask(inputs.size(1)).to(device)
                outputs = model(inputs, mask)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_transformer_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break
            
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses


def main():
    # Hyperparameters
    input_dim = 4  # color, mean, var, next_approximate
    seq_length = 20
    batch_size = 256
    d_model = 512
    nhead = 8
    num_layers = 6
    dropout = 0.2
    learning_rate = 0.0005
    num_epochs = 200
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_loader, val_loader, test_loader, means, stds = prepare_data_from_csv(
        "../data/aviator_dataset_clean.csv", seq_length, batch_size
    )

    model = MultivariateTransformer(input_dim, d_model, nhead, num_layers, dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, 
                                weight_decay=0.01, betas=(0.9, 0.999))
    criterion = nn.HuberLoss()

    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0
    )

    print('Training model...')

    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        num_epochs, device
    )
    print('Training complete')
    print(f'Best Validation Loss: {min(val_losses):.4f}')
    print(f'Best Training Loss: {min(train_losses):.4f}')

    model.load_state_dict(torch.load('best_transformer_model.pth'))
    model.eval()
    
    test_predictions = []
    test_actuals = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            mask = create_attention_mask(inputs.size(1)).to(device)
            with autocast():
                outputs = model(inputs, mask)

            pred = outputs.cpu().numpy() * stds['target'] + means['target']
            actual = targets.numpy() * stds['target'] + means['target']
            
            test_predictions.extend(pred)
            test_actuals.extend(actual)

    test_predictions = np.array(test_predictions)
    test_actuals = np.array(test_actuals)

    mse = np.mean((test_predictions - test_actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(test_predictions - test_actuals))
    mape = np.mean(np.abs((test_actuals - test_predictions) / test_actuals)) * 100

    print('\nTest Metrics:')
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'MAPE: {mape:.2f}%')

    state = {
        'means': means,
        'stds': stds
    }
    with open('model_state.json', 'w') as f:
        json.dump(state, f)

    interactive_prediction(model, means, stds, device)


if __name__ == "__main__":
    main()
