from dataclasses import dataclass


@dataclass
class Config:
    # General options
    device: int = -1
    ollama_url: str = 'http://localhost:11434/v1'
    n_runs: int = 5
    method: str = 'PLMFT'       # 'PLMFT' or 'LLM'
    few_shot: bool = False      # enable 3-shot examples in LLM prompt
    save_results: bool = False  # dump predictions + labels to results.json
    # n_train=-1 means use all training data
    n_train: int = -1
    # n_test=-1 means evaluate on all test data
    n_test: int = -1
