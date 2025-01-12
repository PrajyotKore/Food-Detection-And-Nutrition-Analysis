import os
import torch
import hub
import yaml
import ultralytics
from torchvision import transforms
from typing import Optional, DefaultDict
import numpy as np
import matplotlib.pyplot as plt 
from ultralytics import YOLO
from food_detection_and_nutritional_value.modeling import LLAMA
from food_detection_and_nutritional_value.modeling import YOLOPredict


class train_YOLO:
    def __init__(self, config):
        """
        Initialize YOLO training class
        
        Args:
            model_path (str, optional): Path to a pretrained model or model type like 'yolov8n.pt'
            task (str, optional): Type of task - 'detect', 'segment', or 'classify'
        """

        self.config = config
        self.model_path = self.config['MODEL_PATH']
        self.task= self.config['task']
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_model(self):
        """
        Load YOLO model either from scratch or pretrained weights
        
        Returns:
            YOLO model instance
        """

        if self.model_path:
            # Load existing model
            self.model = YOLO(self.model_path)
            return self.model
        else:
            raise ValueError('Error: No Model path given')
    
    def load_food_data(self, num_workers: Optional[int], shuffle: Optional[bool] , batch_size: Optional[int], type : Optional[str] ):
        # Define transformations
        transform = transforms.Compose([
            # transforms.Resize((128, 128)),               # Resize to 128x128
            transforms.RandomHorizontalFlip(p=0.5),     # 50% chance to flip horizontally
            transforms.RandomRotation(30),             # Random rotation within ±30 degrees
            transforms.ToTensor(),                      # Convert to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5],  # Normalize with mean and std for RGB channels
                                std=[0.5, 0.5, 0.5])   # Standard deviation for RGB channels
        ])
    
        ds = hub.dataset(self.config[type])
        dataloader = ds.pytorch(num_workers = num_workers, shuffle = True, transform = transform, batch_size= batch_size)
        return dataloader

    
    def train(self, data_yaml, epochs=100, imgsz=640, batch_size=16, **kwargs):
        """
        Train the YOLO model
        
        Args:
            data_yaml (str): Path to data.yaml file
            epochs (int): Number of training epochs
            imgsz (int): Input image size
            batch_size (int): Batch size
            **kwargs: Additional training arguments
        
        Returns:
            Training results
        """
        if self.model is None:
            self.load_model()
            
        try:
            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                device=self.device,
                **kwargs
            )
            return results
            
        except Exception as e:
            print(f"Error during training: {str(e)}")


if __name__ == '__main__':
    # DATA_YAML = 'path/to/your/data.yaml'  # path to your data.yaml file
    config_path = 'config/config.yaml'
    with open(config_path) as file:
        config = yaml.safe_load(file)

    # Initialize trainer
    trainer = train_YOLO(config)
    # food_dataloader = trainer.load_food_data(num_workers = 1, shuffle= True, batch_size=2,type = 'train' )
   
    yaml_config_path = 'config\yolo-data.yaml'
    try:
        # Train model
        results = trainer.train(
            data_yaml=yaml_config_path,
            **config['tranining_config']
        )
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed with error: {str(e)}")