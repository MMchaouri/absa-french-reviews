import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW

# Label mapping  
LABEL_MAP = {
    "Positive": 0,
    "Négative": 1,
    "Neutre": 2,
    "NE": 3
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}


# Dataset 
class RestaurantDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["Avis"]

        labels = torch.tensor([
            LABEL_MAP.get(self.data[idx]["Prix"], 3),
            LABEL_MAP.get(self.data[idx]["Cuisine"], 3),
            LABEL_MAP.get(self.data[idx]["Service"], 3)
        ])

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels
        }


# Model 
class PLMFTModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.3)
        self.prix = nn.Linear(hidden, 4)
        self.cuisine = nn.Linear(hidden, 4)
        self.service = nn.Linear(hidden, 4)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        pooled = self.dropout(pooled)

        return {
            "Prix": self.prix(pooled),
            "Cuisine": self.cuisine(pooled),
            "Service": self.service(pooled)
        }


# Classifier wrapper 
class PLMClassifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(
            f"cuda:{cfg.device}" if cfg.device >= 0 and torch.cuda.is_available() else "cpu"
        )
        
        #self.model_name = "distilbert-base-multilingual-cased"
        self.model_name = "camembert-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = PLMFTModel(self.model_name).to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        
        

    # TRAIN 
    def train(self, train_data, val_data, device):
        dataset = RestaurantDataset(train_data, self.tokenizer)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=2e-5)

        self.model.train()
        

        for epoch in range(3):
            total_loss = 0.0
            for batch in loader:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask)

                loss = (
                    self.loss_fn(outputs["Prix"], labels[:, 0]) +
                    self.loss_fn(outputs["Cuisine"], labels[:, 1]) +
                    self.loss_fn(outputs["Service"], labels[:, 2])
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1} | Loss = {total_loss/len(loader):.4f}")

    # PREDICT 
    def predict(self, texts, device):
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=128
                ).to(self.device)

                outputs = self.model(
                    encoding["input_ids"],
                    encoding["attention_mask"]
                )

                pred = {
                    "Prix": ID2LABEL[torch.argmax(outputs["Prix"]).item()],
                    "Cuisine": ID2LABEL[torch.argmax(outputs["Cuisine"]).item()],
                    "Service": ID2LABEL[torch.argmax(outputs["Service"]).item()]
                }

                predictions.append(pred)

        return predictions
