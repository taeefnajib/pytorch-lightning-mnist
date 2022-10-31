### !pip install --quiet "pandas" "ipython[notebook]" "torchvision" "setuptools==59.5.0" "torch>=1.8" "torchmetrics>=0.7" "seaborn" "pytorch-lightning>=1.4"

### Import all dependencies
import os
import pandas as pd
import seaborn as sn
import torch
from IPython.core.display import display
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
from dataclasses import dataclass

@dataclass
class Hyperparameters(object):
    """
    
    """
    hidden_size=64
    learning_rate = 2e-4
    PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
    BATCH_SIZE = 256 if torch.cuda.is_available() else 64
    norm1 = 0.1307
    norm2 = 0.3081
    

hp = Hyperparameters()

class LitMNIST(LightningModule):
    def __init__(self, data_dir=hp.PATH_DATASETS,
                 hidden_size=hp.hidden_size,
                 learning_rate=hp.learning_rate,
                 norm1 = hp.norm1,
                 norm2 = hp.norm2):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((norm1,), (norm2,)),
            ]
        )

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, x):
        return F.log_softmax(self.model(x), dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return F.nll_loss(self(x), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)
        
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        return MNIST(self.data_dir, train=True, download=True), MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=hp.BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=hp.BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=hp.BATCH_SIZE)

model = LitMNIST()

### Training model
def train_model(model):
    trainer = Trainer(accelerator="auto",
                      devices=1 if torch.cuda.is_available() else None, 
                      max_epochs=3, callbacks=[TQDMProgressBar(refresh_rate=20)], 
                      logger=CSVLogger(save_dir="logs/"),)
                      
    trainer.fit(model)
    return trainer

trainer = train_model(model)

### Testing model
def test_model(trainer, model):
    return trainer.test(model)

test_model(trainer, model)