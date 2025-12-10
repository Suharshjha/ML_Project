from models.med import BLIP
from training.train_pretrain import train_pretrain
from data.dataset_web import WebDataset
from torch.utils.data import DataLoader
import torch.optim as optim

def main():
    model = BLIP()
    dataset = WebDataset("/path/to/data")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(20):
        train_pretrain(model, loader, optimizer)

if __name__ == "__main__":
    main()
