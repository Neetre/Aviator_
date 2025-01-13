import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.optim.lr_scheduler import StepLR

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def prepare_data_from_csv(file_path, seq_length):
    df = pd.read_csv(file_path)
    multipliers = df["Multiplier"].values.astype(np.float32)

    mean = np.mean(multipliers)
    std = np.std(multipliers)
    multipliers = (multipliers - mean) / std

    data = []
    for i in range(len(multipliers) - seq_length):
        sequence = multipliers[i:i+seq_length]
        data.append(sequence)

    data = np.array(data, dtype=np.float32)
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(-1)
    return data, mean, std


input_size = 1
hidden_size = 32
num_layers = 2
output_size = 1
seq_length = 10
batch_size = 32
learning_rate = 0.01
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file_path = "../data/multipliers.csv"
data, data_mean, data_std = prepare_data_from_csv(file_path, seq_length)
data = data.to(device)

train_size = int(0.8 * len(data))
train_data = data[:train_size]
val_data = data[train_size:]

model = RNN(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(train_data), batch_size):
        inputs = train_data[i:i+batch_size, :-1, :]
        targets = train_data[i:i+batch_size, -1, :].reshape(-1, 1)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    scheduler.step()

    model.eval()
    with torch.no_grad():
        val_inputs = val_data[:, :-1, :]
        val_targets = val_data[:, -1, :].reshape(-1, 1)
        val_outputs = model(val_inputs)
        val_loss = criterion(val_outputs, val_targets)
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

model.eval()
test_sequence = val_data[0, :-1, :].unsqueeze(0)
predicted_value = model(test_sequence)

predicted_value_denorm = predicted_value.cpu().item() * data_std + data_mean
actual_value_denorm = val_data[0, -1, :].cpu().item() * data_std + data_mean

input_sequence_np = test_sequence.cpu().numpy().flatten()
input_sequence_denorm = input_sequence_np * data_std + data_mean

plt.figure(figsize=(10, 6))
plt.plot(input_sequence_denorm, marker='o', label='Input Sequence')
plt.plot([len(input_sequence_denorm)], [actual_value_denorm], marker='x', markersize=10, color='green', label='Actual Value')
plt.plot([len(input_sequence_denorm)], [predicted_value_denorm], marker='*', markersize=10, color='red', label='Predicted Value')

plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Sequence Prediction')
plt.legend()
plt.grid(True)

plt.show()

print(f"Input Sequence: {input_sequence_denorm}")
print(f"Predicted Value: {predicted_value_denorm}")
print(f"Actual Value: {actual_value_denorm}")