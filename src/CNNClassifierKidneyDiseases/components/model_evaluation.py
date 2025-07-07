import mlflow.pytorch
import mlflow.pytorch
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from urllib.parse import urlparse
from pathlib import Path
from src.CNNClassifierKidneyDiseases.utils.common import save_json
from src.CNNClassifierKidneyDiseases.entity.config_entity import EvaluationConfig
import dagshub
#from src.CNNClassifierKidneyDiseases.entity.config_entity import EvaluationConfig

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_loader(self):
        transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size[1:]),
            transforms.ToTensor()
        ])

        full_dataset = datasets.ImageFolder(root=self.config.training_data, transform=transform)

        # Split dataset into train and validation
        val_size = int(0.3 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        _, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

        self.valid_loader = DataLoader(
            val_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False
        )

    @staticmethod
    def load_model(path: Path):
        return torch.load(path , weights_only=False)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self.model.eval()

        self._valid_loader()

        correct = 0
        total = 0
        loss_total = 0.0
        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        with torch.no_grad():
            for images, labels in self.valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss_total += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        avg_loss = loss_total / total
        self.score = (avg_loss, accuracy)

        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        dagshub.init(
        repo_owner='omarkhadrawy10',
        repo_name='Kidney-Disease-Classification',
        mlflow=True
    )
        mlflow.set_registry_uri(self.config.mlflow_uri)
        #mlflow.set_tracking_uri(self.config.mlflow_uri)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})

            if tracking_url_type_store != "file":
                mlflow.pytorch.log_model(self.model, artifact_path ="model")
            else:
                mlflow.pytorch.log_model(self.model, artifact_path = "model")
