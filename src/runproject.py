import json
import time
from statistics import mean

import pandas as pd
from pprint import pprint
import pyrallis
from lightning import seed_everything
from sklearn.metrics import f1_score

from config import Config
from classifier_wrapper import ClassifierWrapper

ASPECTS = ['Prix', 'Cuisine', 'Service']
LABELS = ['Positive', 'Négative', 'Neutre', 'NE']


def load_data():
    df_train = pd.read_csv("../data/ftdataset_train.tsv", sep=' *\t *', encoding='utf-8', engine='python')
    df_val = pd.read_csv("../data/ftdataset_val.tsv", sep=' *\t *', encoding='utf-8', engine='python')
    try:
        df_test = pd.read_csv("../data/ftdataset_test.tsv", sep=' *\t *', encoding='utf-8', engine='python')
    except Exception:
        df_test = df_val
    return df_train.to_dict(orient='records'), df_val.to_dict(orient='records'), df_test.to_dict(orient='records')


def eval(preds: list[dict], test_data: list[dict]) -> dict:
    n = len(test_data)
    results = {}

    for aspect in ASPECTS:
        y_true = [row[aspect] for row in test_data]
        y_pred = [p[aspect] if p else 'NE' for p in preds]

        correct = sum(t == p for t, p in zip(y_true, y_pred))
        acc = round(100 * correct / n, 2)
        f1 = round(100 * f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0), 2)

        results[aspect] = {'acc': acc, 'f1': f1}

    macro_acc = round(mean(r['acc'] for r in results.values()), 2)
    macro_f1 = round(mean(r['f1'] for r in results.values()), 2)
    results['macro_acc'] = macro_acc
    results['macro_f1'] = macro_f1
    return results


def run_project(cfg: Config):
    train_data, val_data, test_data = load_data()
    if cfg.n_train > 0:
        train_data = train_data[:cfg.n_train]
    if cfg.n_test > 0:
        test_data = test_data[:cfg.n_test]

    test_texts = [row['Avis'] for row in test_data]

    if cfg.method == 'LLM':
        cfg.n_runs = 1

    pprint(vars(cfg), sort_dicts=False, compact=True)

    all_runs_acc = []
    all_runs_f1 = []
    last_preds = None

    for run_id in range(1, cfg.n_runs + 1):
        print(f"\nRUN {run_id}/{cfg.n_runs}")
        wrapper = ClassifierWrapper(cfg)
        if cfg.method == 'PLMFT':
            wrapper.train(train_data, val_data, cfg.device)

        print("Evaluating...")
        preds = wrapper.predict(test_texts, cfg.device)
        metrics = eval(preds, test_data)
        last_preds = preds

        print(f"\nRUN {run_id} results:")
        for aspect in ASPECTS:
            print(f"  {aspect:10s}  acc={metrics[aspect]['acc']:.1f}%  f1={metrics[aspect]['f1']:.1f}%")
        print(f"  {'macro':10s}  acc={metrics['macro_acc']:.1f}%  f1={metrics['macro_f1']:.1f}%")

        all_runs_acc.append(metrics['macro_acc'])
        all_runs_f1.append(metrics['macro_f1'])

    print(f"\nALL RUNS — macro acc: {all_runs_acc}  avg: {round(mean(all_runs_acc), 2)}")
    print(f"ALL RUNS — macro  f1: {all_runs_f1}  avg: {round(mean(all_runs_f1), 2)}")

    if cfg.save_results and last_preds is not None:
        output = {
            "method": cfg.method,
            "few_shot": cfg.few_shot,
            "predictions": last_preds,
            "ground_truth": [{a: row[a] for a in ASPECTS} for row in test_data],
        }
        with open("../results.json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print("Results saved to results.json")


if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=Config)
    seed_everything(123)
    start_time = time.perf_counter()
    run_project(cfg)
    elapsed = round(time.perf_counter() - start_time, 1)
    print(f"\nTotal time: {elapsed}s")
