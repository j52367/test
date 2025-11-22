import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.utils import seed_everything
from src.dataset import CustomDataset
from src.model import BaseModel
from src.trainer import train_model

def main(args):
    # 1. Seed Setting
    seed_everything(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using Device: {device}")

    # 2. Data Loading
    print("Loading Data...")
    train_data = pd.read_csv('./data/train.csv').drop(columns=['ID', '제품'])

    # 3. Preprocessing (Scaling)
    print("Preprocessing Data...")
    scale_max_dict = {}
    scale_min_dict = {}

    for idx in range(len(train_data)):
        maxi = np.max(train_data.iloc[idx, 4:])
        mini = np.min(train_data.iloc[idx, 4:])

        if maxi == mini:
            train_data.iloc[idx, 4:] = 0
        else:
            train_data.iloc[idx, 4:] = (train_data.iloc[idx, 4:] - mini) / (maxi - mini)

        scale_max_dict[idx] = maxi
        scale_min_dict[idx] = mini

    # 4. Label Encoding
    label_encoder = LabelEncoder()
    categorical_columns = ['대분류', '중분류', '소분류', '브랜드']

    for col in categorical_columns:
        label_encoder.fit(train_data[col])
        train_data[col] = label_encoder.transform(train_data[col])

    # 5. Dataset & DataLoader
    dataset = CustomDataset(train_data, args.train_window_size, args.predict_size)
    
    total_size = len(dataset)
    train_size = int(total_size * 0.8)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 6. Model & Optimizer
    model = BaseModel(input_size=5, hidden_size=args.hidden_size, output_size=args.predict_size)
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    # 7. Training
    print("Start Training...")
    train_model(model, optimizer, train_loader, val_loader, device, args.epochs, args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--train_window_size', type=int, default=90)
    parser.add_argument('--predict_size', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--save_path', type=str, default='./saved_models')

    args = parser.parse_args()
    main(args)