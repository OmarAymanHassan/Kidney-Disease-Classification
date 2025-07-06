import os
from zipfile import ZipFile
import urllib.request as requests
import timm
import torch.nn as nn
from torch.nn import Module
import torch.optim as optim
from typing import Optional
from pathlib import Path
import tqdm 
import torch 
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from src.CNNClassifierKidneyDiseases.entity.config_entity import TrainingConfig

from src.CNNClassifierKidneyDiseases import logger

class Training:

    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = torch.load(self.config.updated_base_model, weights_only=False)

    def train_valid_loader(self):
        image_size = self.config.params_image_size[1:]
        means = [0.1921, 0.1921, 0.1921]
        stds = [0.2601, 0.2601, 0.2601]

        if self.config.params_is_augmented:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=30),
                transforms.ToTensor(),
                transforms.Normalize(mean=means, std=stds)
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=means, std=stds)
            ])

        valid_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds)
        ])

        full_dataset = datasets.ImageFolder(self.config.training_data, transform=train_transform)
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size

        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.params_batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.config.params_batch_size, shuffle=False)

    @staticmethod
    def save_model(path: Path, model: torch.nn.Module):
        torch.save(model, path)

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        logger.info(f"Training started on device: {device}")

        for epoch in range(self.config.params_epochs):
            self.model.train()
            total_loss, correct, total = 0, 0, 0

            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                y_pred = self.model(images)
                loss = criterion(y_pred, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, preds = torch.max(y_pred, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                # Log loss for each batch
                """logger.info(f"Epoch [{epoch+1}/{self.config.params_epochs}] - "
                            f"Batch [{batch_idx+1}/{len(self.train_loader)}] - "
                            f"Batch Loss: {loss.item():.4f}") """

            train_accuracy = correct / total
            logger.info(f"Epoch [{epoch+1}/{self.config.params_epochs}] - "
                        f"Total Loss: {total_loss:.4f} - Accuracy: {train_accuracy:.4f}")

        # Validation
        self.model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in self.valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / val_total
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")

        self.save_model(path=self.config.trained_model_path, model=self.model)
        logger.info(f"Model saved to: {self.config.trained_model_path}")
