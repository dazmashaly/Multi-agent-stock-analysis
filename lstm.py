import torch
from torch import nn
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset


@dataclass
class LSTMSelfConfig:
  hidden_dim: int = 256
  num_heads: int = 4
  num_blocks: int = 2
  attn_drop: float = 0.1
  qkv_bias: bool = True
  dropout: float = 0.1
  act: str = "ReLU"
  in_dims: int = 1

  def as_dict(self):
    return {
        "hidden_dim": self.hidden_dim,
        "num_heads": self.num_heads,
        "num_blocks": self.num_blocks,
        "attn_drop": self.attn_drop,
        "qkv_bias":self.qkv_bias,
       "dropout":self.dropout,
        "act":self.act,
        "in_dims":self.in_dims,
    }

class SelfAttention(nn.Module):
  def __init__(self, config):
      super(SelfAttention, self).__init__()
      self.wqkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, config.qkv_bias)
      self.wo = nn.Linear(config.hidden_dim, config.hidden_dim, config.qkv_bias)
      self.num_heads = config.num_heads
      self.head_dim = config.hidden_dim // self.num_heads
      self.attn_drop = config.attn_drop
      self.hidden_dim = config.hidden_dim

  def forward(self, hidden_states):
      bs = hidden_states.size(0)
      qkv = self.wqkv(hidden_states)
      q, k, v = torch.chunk(qkv, 3, dim=-1)
      q = q.view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
      k = k.view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
      v = v.view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
      attn_drop = self.attn_drop if self.training else 0.0
      attention_scores = nn.functional.scaled_dot_product_attention(
          q, k, v, dropout_p=attn_drop)

      attention_scores = attention_scores.permute(0, 2, 1, 3).reshape(bs, -1, self.num_heads * self.head_dim)
      return self.wo(attention_scores)


class LSTMSelfBlock(nn.Module):
  def __init__(self, in_dims,config):
    super(LSTMSelfBlock, self).__init__()
    self.config = config
    self.lstm = nn.LSTM(in_dims, config.hidden_dim, batch_first=True)
    self.self_attention = SelfAttention(config)
    self.act = getattr(nn, config.act)()
    self.dropout = nn.Dropout(config.dropout)
    self.layernorm = nn.LayerNorm(config.hidden_dim)
  
  def forward(self, inputs):
      out, _ = self.lstm(inputs)
      out = self.layernorm(out)
      out = out + self.dropout(self.self_attention(out))
      return self.act(out)



class LSTMSelf(nn.Module):
  def __init__(self, config):
    super(LSTMSelf, self).__init__()
    self.config = config
    in_dims = config.in_dims
    self.blocks = nn.ModuleList([])
    for i in range(config.num_blocks):
      self.blocks.append(LSTMSelfBlock(in_dims, config))
      in_dims = config.hidden_dim
    self.drop = nn.Dropout(config.dropout)
    self.fc = nn.Linear(config.hidden_dim, 1)
  
  def forward(self, inputs):
      x = inputs
      for blc in self.blocks:
        x = blc(x)
      x = self.drop(x)
      out = self.fc(x)
      return out


class StockDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data.copy()
        self.seq_len = seq_len
        self.organize_data()

    def __len__(self):
        return len(self.data)

    def organize_data(self):
       self.data = self.data.sort_values(by="date")
       self.data = self.data['price'].values
       data = []
       targets = []
       for st in range(len(self.data)):
          if st + self.seq_len >= len(self.data):
            break
          data.append(self.data[st:st+self.seq_len])
          targets.append(self.data[st+self.seq_len])
       self.data = data
       self.targets = targets

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx][...,None]).to(torch.float32)
        y = torch.tensor(self.targets[idx]).to(torch.float32)
        return x, y


def rmse(y_true, y_pred):
    return torch.sqrt(nn.functional.mse_loss(y_pred, y_true)).item()


def r2(y_true, y_pred):
    return r2_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())


def fit_lstm(model, train_loader, val_loader=None, num_epochs=10, learning_rate=0.0001, logging=False):
    # Loss function and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        # TQDM progress bar
        progress_bar = train_loader
        if logging:
           progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100, leave=False)
        
        for inputs, targets in progress_bar:
            # Send data to device (GPU or CPU)
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            outputs = outputs.squeeze(-1)[:, 0]

            # Compute loss
            #print(outputs.shape, targets.shape, inputs.shape)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track the loss
            running_loss += loss.item()
            
            # Store predictions and targets for metrics
            all_preds.append(outputs)
            all_labels.append(targets)
            if logging:
               progress_bar.set_postfix(loss=running_loss / (len(all_preds)))

        # Calculate RMSE and R2 on training data
        preds = torch.cat(all_preds, dim=0).detach().cpu()
        labels = torch.cat(all_labels, dim=0).detach().cpu()
        #print(labels.shape, preds.shape)
        train_rmse = rmse(labels, preds)
        train_r2 = r2(labels, preds)

        # Validation step
        if val_loader is not None:
          model.eval()
          val_preds, val_labels = [], []
          with torch.no_grad():
              for inputs, targets in val_loader:
                  inputs, targets = inputs.to(device), targets.to(device)
                  outputs = model(inputs)
                  outputs = outputs.squeeze(-1)[:, 0]
                  val_preds.append(outputs)
                  val_labels.append(targets)

          # Concatenate validation predictions and targets
          val_preds = torch.cat(val_preds, dim=0).detach().cpu()
          val_labels = torch.cat(val_labels, dim=0).detach().cpu()

          # Calculate RMSE and R2 on validation data
          val_rmse = rmse(val_labels, val_preds)
          val_r2 = r2(val_labels, val_preds)

        if logging:
          # Print epoch summary
          print(f"Epoch {epoch + 1}/{num_epochs}:")
          print(f"  Training Loss: {running_loss / len(train_loader):.4f}")
          print(f"  Training RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}")
          if val_loader is not None:
             print(f"  Validation RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}")
          print('-' * 40)


class LSTMModelHandler:
   def __init__(self, model_json_path):
       json_config = json.loads(open(model_json_path,"rb").read())  
       self.modelConfig = LSTMSelfConfig(**json_config)
       self.model = LSTMSelf(self.modelConfig)

   def fit(self, data, num_epochs=10, batch_size=16, logging=False, input_seq_len=10):
        train_loader = DataLoader(StockDataset(data, input_seq_len), batch_size=batch_size, shuffle=True)
        fit_lstm(self.model, train_loader, num_epochs=num_epochs, logging=logging)  
   
   def predict(self, data, input_seq_len=10):
       self.model.eval()
       dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       self.model = self.model.to(dev)
       if isinstance(data, pd.DataFrame) and "date" in data.columns:
          data = data.sort_values(by="date")
          data = data['price'].values
     
       data = torch.from_numpy(data[-input_seq_len:].astype(np.float32))[None, ...]
       data = data.unsqueeze(-1).to(torch.float32).to(dev)
       with torch.no_grad():
           out = self.model(data).squeeze(-1)
           out = (out[:, 0]).squeeze().item()
       return out 
   
   def fit_predict(self, data, num_epochs=10, batch_size=16, logging=False, input_seq_len=10):
       self.fit(data, num_epochs, batch_size, logging, input_seq_len)
       return self.predict(data, input_seq_len)
