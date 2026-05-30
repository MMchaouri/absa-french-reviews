from tqdm import tqdm

from config import Config
from llm_classifier import LLMClassifier
from plmft_classifier import PLMClassifier


class ClassifierWrapper:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.method = cfg.method
        if self.method == 'PLMFT':
            self.classifier = PLMClassifier(cfg)
        else:
            self.classifier = LLMClassifier(cfg)

    def train(self, train_data: list[dict], val_data: list[dict], device: int) -> None:
        if self.method == 'PLMFT':
            print("Training PLMFT model...")
            self.classifier.train(train_data, val_data, device)

    def predict(self, texts: list[str], device: int) -> list[dict]:
        all_opinions = []
        if self.method == 'LLM':
            for text in tqdm(texts):
                opinions = self.classifier.predict(text)
                all_opinions.append(opinions)
        else:
            for text in tqdm(texts):
                opinions = self.classifier.predict([text], device)[0]
                all_opinions.append(opinions)
        return all_opinions
