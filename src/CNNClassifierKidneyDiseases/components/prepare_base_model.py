import os
from zipfile import ZipFile
import torch 
import urllib.request as requests
import timm
import tqdm 
import torch.nn as nn
from torch.nn import Module
import torch.optim as optim
from typing import Optional
from pathlib import Path

from src.CNNClassifierKidneyDiseases.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self,config:PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = timm.create_model(self.config.params_model_name, pretrained=True)

        self.save_model(path=self.config.base_model_path , model=self.model)
    

    @staticmethod
    def _prepare_full_model(model: Module, classes: int, learning_rate: float):
        """
        Equivalent to the Keras _prepare_full_model function.
        - Freezes layers
        - Adds a classifier head
        - Prepares optimizer and loss
        """
        # Freeze all or part of the model
        layers = list(model.children())

        for param in model.parameters():
            param.requires_grad = False



        # Get output features from the model
        # classif : is the last layer of this model
        in_features = model.classif.in_features


        # Replace the classifier head
        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, classes)
        )

        # Attach new classifier
        model.classif = classifier


        # ✅ Unfreeze classifier parameters
        for param in classifier.parameters():
            param.requires_grad = True

        # Set optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Print model summary (simplified)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params}/{total_params}")

        return model, criterion, optimizer

    def update_base_model(self):
        self.full_model, self.criterion, self.optimizer = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)



    @staticmethod
    def save_model(path:Path , model:torch.nn.Module):
        torch.save(model,path)

    