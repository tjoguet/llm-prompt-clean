# src/training/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import os
from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ComplexityDataset(Dataset):
    """Dataset personnalisé pour les données de complexité."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """Charge les données depuis un fichier JSONL."""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenisation
        encoding = self.tokenizer(
            item['query'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'complexity_score': torch.tensor(item['complexity_score'], dtype=torch.float)
        }

class ComplexityTrainer:
    """Trainer pour le modèle ComplexityScorer."""
    
    def __init__(self, 
                 model,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 batch_size: int = 32,
                 learning_rate: float = 2e-5,
                 num_epochs: int = 10,
                 save_dir: str = 'models/checkpoints'):
        
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        
        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        # Optimizer et loss
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Loss pour une seule sortie
        self.criterion = nn.MSELoss()
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Créer le dossier de sauvegarde
        os.makedirs(save_dir, exist_ok=True)
    
    def train_epoch(self) -> float:
        """Entraîne le modèle sur une époque."""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            # Déplacer sur device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            target_score = batch['complexity_score'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_score = self.model(input_ids, attention_mask)
            
            # Calcul de la loss
            loss = self.criterion(pred_score, target_score)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Valide le modèle."""
        self.model.eval()
        total_loss = 0
        all_pred_scores = []
        all_target_scores = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Déplacer sur device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                target_score = batch['complexity_score'].to(self.device)
                
                # Forward pass
                pred_score = self.model(input_ids, attention_mask)
                
                # Loss
                loss = self.criterion(pred_score, target_score)
                total_loss += loss.item()
                
                # Collecter pour métriques
                all_pred_scores.extend(pred_score.cpu().numpy())
                all_target_scores.extend(target_score.cpu().numpy())
        
        # Calcul des métriques
        score_mse = mean_squared_error(all_target_scores, all_pred_scores)
        score_mae = mean_absolute_error(all_target_scores, all_pred_scores)
        
        metrics = {
            'score_mse': score_mse,
            'score_mae': score_mae
        }
        
        return total_loss / len(self.val_loader), metrics
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        """Sauvegarde un checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Sauvegarde du checkpoint courant
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Sauvegarde du meilleur modèle
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"✅ Nouveau meilleur modèle sauvegardé ! Val Loss: {val_loss:.4f}")
    
    def train(self):
        """Boucle d'entraînement principale."""
        print(f"🚀 Début de l'entraînement sur {self.device}")
        print(f"📊 Dataset: {len(self.train_loader)} batches train, {len(self.val_loader)} batches val")
        
        for epoch in range(self.num_epochs):
            print(f"\n📈 Époque {epoch+1}/{self.num_epochs}")
            
            # Entraînement
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, metrics = self.validate()
            self.val_losses.append(val_loss)
            
            # Affichage des métriques
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Score MAE: {metrics['score_mae']:.4f}")
            
            # Sauvegarde
            self.save_checkpoint(epoch + 1, val_loss)
        
        print(f"\n🎉 Entraînement terminé !")
        print(f"Meilleur modèle: Val Loss = {self.best_val_loss:.4f}")