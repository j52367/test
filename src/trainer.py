import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import os

def validation(model, val_loader, criterion, device):
    model.eval()
    val_loss = []

    with torch.no_grad():
        for X, Y in tqdm(iter(val_loader), desc="Validating", leave=False):
            X = X.float().to(device)
            Y = Y.float().to(device)

            output = model(X)
            loss = criterion(output, Y)

            val_loss.append(loss.item())
    return np.mean(val_loss)

def train_model(model, optimizer, train_loader, val_loader, device, epochs, save_path):
    model.to(device)
    criterion = nn.MSELoss().to(device)
    best_loss = 9999999
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(1, epochs+1):
        model.train()
        train_loss = []
        
        progress_bar = tqdm(iter(train_loader), desc=f"Epoch {epoch}/{epochs}")
        for X, Y in progress_bar:
            X = X.float().to(device)
            Y = Y.float().to(device)

            optimizer.zero_grad()

            output = model(X)
            loss = criterion(output, Y)

            loss.backward()
            optimizer.step()\

            train_loss.append(loss.item())
            progress_bar.set_postfix({'loss': np.mean(train_loss)})

        val_loss = validation(model, val_loader, criterion, device)
        print(f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}]')

        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
            print('Model Saved')