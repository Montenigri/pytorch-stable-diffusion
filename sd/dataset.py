from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

class Dataset(Dataset):

    def __init__(self, csv_file):
        self.csv = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.csv)

    def get_batch_images(self, idx):
        input = Image.open(self.csv.iloc[idx, 0])
        target = Image.open(self.csv.iloc[idx, 1])
        context = self.csv.iloc[idx, 2]
        return input, target, context
    
    def __getitem__(self, idx):
        item = self.get_batch_images(idx)
        return item
    
def create_dataloaders(csv_file, batch_size=4):
    # Read the CSV file
    data = pd.read_csv(csv_file)
    
    # Split the data into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Save the split data into temporary CSV files
    train_csv = 'train_split.csv'
    test_csv = 'test_split.csv'
    train_data.to_csv(train_csv, index=False)
    test_data.to_csv(test_csv, index=False)
    
    # Create Dataset instances
    train_dataset = Dataset(train_csv)
    test_dataset = Dataset(test_csv)
    
    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader