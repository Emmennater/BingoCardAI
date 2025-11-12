import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from helper import random_transform
from tqdm import tqdm

# -------------------------
# Model Definition
# -------------------------
class CornerNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = nn.Sequential(
      nn.Conv2d(1, 32, 3, stride=2, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),

      nn.Conv2d(32, 64, 3, stride=2, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),

      nn.Conv2d(64, 128, 3, stride=2, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),

      nn.Conv2d(128, 256, 3, stride=2, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(),

      nn.Conv2d(256, 512, 3, stride=2, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(),
    )

    self.head = nn.Sequential(
      nn.Flatten(),
      nn.Linear(512 * 13 * 13, 1024),
      nn.ReLU(),
      nn.Linear(1024, 8),
    )

  def forward(self, x):
    x = self.encoder(x)
    return self.head(x)

  @staticmethod
  def save(model, path):
    print("Saving model to", path)
    torch.save(model.state_dict(), path + ".tmp")
    os.replace(path + ".tmp", path)

  @staticmethod
  def load(path):
    try:
      model = CornerNet()
      model.load_state_dict(torch.load(path))
      print("Model loaded from", path)
      return model
    except FileNotFoundError:
      print("Creating new model")
      return CornerNet()

# -------------------------
# Dataset Definition
# -------------------------
class BingoCornerDataset(Dataset):
  def __init__(self, img, n_samples=5000):
    self.img = img
    self.n_samples = n_samples

  def __len__(self):
    return self.n_samples

  def __getitem__(self, idx):
    warped, true_corners = random_transform(self.img)

    # ensure shape (H, W, 1)
    if warped.ndim == 2:
      warped = warped[..., None]

    warped = torch.from_numpy(warped).permute(2, 0, 1).float() / 255.0  # (1, H, W)
    true_corners = torch.from_numpy(true_corners).float() / 400.0       # normalize coords

    return warped, true_corners

# -------------------------
# Training Loop
# -------------------------
def train():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  img = cv2.imread("input.png", cv2.IMREAD_GRAYSCALE)
  img = cv2.resize(img, (400, 400))

  dataset = BingoCornerDataset(img, n_samples=5000)
  dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

  model = CornerNet.load("model.pt").to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  loss_fn = nn.MSELoss()

  log_interval = 10
  running_loss = 0.0

  with tqdm(dataloader, desc="Training", unit="batch") as prog_bar:
    for i, (warped_batch, corner_batch) in enumerate(prog_bar, 1):
      warped_batch = warped_batch.to(device)
      corner_batch = corner_batch.to(device)

      optimizer.zero_grad()
      preds = model(warped_batch)
      loss = loss_fn(preds, corner_batch)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

      # update the tqdm description every n iterations
      if i % log_interval == 0:
        avg_loss = running_loss / log_interval
        prog_bar.set_description(f"Avg Loss: {avg_loss:.6f}")
        running_loss = 0.0
        CornerNet.save(model, "model.pt")

if __name__ == "__main__":
  train()
