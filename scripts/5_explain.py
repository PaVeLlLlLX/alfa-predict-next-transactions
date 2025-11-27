import sys
import os
from typing import Union, List, Any, Optional
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import shap
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.amp import autocast
from src import config, dataset, model, utils
import gc

plt.rcParams.update({'figure.autolayout': True})

logger = utils.get_logger("explainability")

EXPLAIN_BATCH_SIZE = 256 
BACKGROUND_SAMPLES = 30   
EXPLAIN_SAMPLES = 100     
NSAMPLES_SHAP = 256 

class GpuBatchedWrapper:
    def __init__(self, model: nn.Module, device: Union[str, torch.device], n_num_features: int, batch_size: int):
        self.model = model
        self.device = device
        self.n_num = n_num_features
        self.batch_size = batch_size
        self.model.eval()

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        n_samples = X.shape[0]
        all_probs: List[np.ndarray] = []
        
        with torch.no_grad():
            for i in range(0, n_samples, self.batch_size):
                batch_X = X[i : i + self.batch_size]
                
                x_num_np = batch_X[:, :self.n_num].astype(np.float32)
                x_cat_np = np.round(batch_X[:, self.n_num:]).astype(np.int64)

                x_num = torch.tensor(x_num_np).to(self.device)
                x_cat = torch.tensor(x_cat_np).to(self.device)
                
                with autocast("cuda"):
                    logits = self.model(x_num, x_cat)
                    probs = torch.sigmoid(logits)
                
                all_probs.append(probs.float().cpu().numpy())
        
        return np.vstack(all_probs)
   

class SklearnModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model: nn.Module, device: Union[str, torch.device], n_num_features: int):
        self.model = model
        self.device = device
        self.n_num = n_num_features
        self.model.eval()
    # Заглушка
    def fit(self, X: Any, y: Any = None) -> 'SklearnModelWrapper':
        return self

    def predict(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        x_num_np = X[:, :self.n_num].astype(np.float32)
        x_cat_np = X[:, self.n_num:].astype(np.int64)

        with torch.no_grad():
            x_num = torch.tensor(x_num_np).to(self.device)
            x_cat = torch.tensor(x_cat_np).to(self.device)
            
            logits = self.model(x_num, x_cat)
            probs = torch.sigmoid(logits).cpu().numpy()
            
        return probs

def main() -> None:
    utils.seed_everything(config.SEED)
    
    logger.info("Загрузка ресурсов")
    svd = joblib.load(config.SVD_PATH)
    mcc_map = pd.read_excel(config.MCC_MAP_PATH)
    mcc_dict = dict(zip(mcc_map["mcc"].astype(str), mcc_map["eng_cat"]))
    with open(config.VOCAB_PATH, 'r') as f: vocabs = json.load(f)
    
    encoder = dataset.CategoricalEncoder(vocabs)
    processor = dataset.DataProcessor(svd, mcc_dict)
    cards = [len(vocabs[c]) for c in config.CAT_COLS]
    
    calc_device = torch.device('cuda')
    logger.info(f"Используется устройство для расчетов: {calc_device}")
    
    ftt = model.MultiLabelFTTransformer(len(config.NUM_COLS), cards).to(calc_device)
    ftt.load_state_dict(torch.load(config.MODEL_CHECKPOINT, map_location=calc_device))
    
    logger.info("Подготовка данных...")
    val_ds = dataset.TabularStreamingDataset([config.VAL_FINAL], encoder, processor)
    val_loader = DataLoader(val_ds, batch_size=EXPLAIN_BATCH_SIZE, num_workers=0) 
    
    batch = next(iter(val_loader))
    x_num = batch['x_num'].numpy()
    x_cat = batch['x_cat'].numpy()
    y_true = batch['labels'].numpy()
    
    X_sample = np.hstack([x_num, x_cat])
    feature_names = config.NUM_COLS + config.CAT_COLS
    logger.info(f"Признаков: {len(feature_names)}")

    wrapper_sklearn = SklearnModelWrapper(ftt, calc_device, len(config.NUM_COLS))

    logger.info("\n  Запуск Permutation Importance")
    
    def multi_label_scorer(estimator: Any, X: np.ndarray, y: np.ndarray) -> float:
        probs = estimator.predict(X)
        preds = (probs >= 0.4).astype(int)
        return f1_score(y, preds, average='micro')

    pi_result = permutation_importance(
        wrapper_sklearn, X_sample, y_true, 
        scoring=multi_label_scorer, 
        n_repeats=5, 
        random_state=42, 
        n_jobs=1,
    )
    del y_true, val_ds, val_loader, wrapper_sklearn
    gc.collect()
    torch.cuda.empty_cache()

    sorted_idx = pi_result.importances_mean.argsort()
    
    plt.figure(figsize=(12, 10))
    plt.boxplot(
        pi_result.importances[sorted_idx].T,
        vert=False,
        labels=np.array(feature_names)[sorted_idx]
    )
    plt.title("Permutation Importance (F1 Micro)")
    plt.tight_layout()
    plt.savefig(f"{config.ARTIFACTS_DIR}/permutation_importance.png")
    logger.info("Permutation Importance сохранен")

    del pi_result, sorted_idx
    gc.collect()

    logger.info("Запуск SHAP")

    wrapper_shap = GpuBatchedWrapper(
        ftt, 
        config.DEVICE, 
        n_num_features=len(config.NUM_COLS),
        batch_size=EXPLAIN_BATCH_SIZE
    )

    logger.info(f"Генерация фона (KMeans, {BACKGROUND_SAMPLES} точек)...")
    bg_subset = X_sample[np.random.choice(X_sample.shape[0], min(1000, X_sample.shape[0]), replace=False)]
    bg_df = pd.DataFrame(bg_subset, columns=feature_names)
    
    background_data = shap.kmeans(bg_df, BACKGROUND_SAMPLES)
    explainer = shap.KernelExplainer(wrapper_shap.predict, background_data)
    
    X_explain = X_sample[:EXPLAIN_SAMPLES]
    X_explain_df = pd.DataFrame(X_explain, columns=feature_names)
    
    logger.info(f"Запуск расчета SHAP для {EXPLAIN_SAMPLES} примеров...")
    shap_values = explainer.shap_values(X_explain_df, nsamples=NSAMPLES_SHAP)
    
    logger.info("Обработка результатов")
    
    if isinstance(shap_values, list):
        shap_values_np = np.array(shap_values).transpose(1, 2, 0)
    else:
        shap_values_np = np.array(shap_values)
        
    combined_shap_values = np.sum(np.abs(shap_values_np), axis=2)
    
    logger.info(f"Combined shape for Bar plot: {combined_shap_values.shape}")
    
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        combined_shap_values, 
        X_explain_df, 
        plot_type="bar", 
        show=False,
        max_display=20
    )
    plt.title("Global Feature Importance (Sum across all classes)")
    plt.tight_layout()
    plt.savefig(f"{config.ARTIFACTS_DIR}/shap_global_bar.png")
    plt.close()
    logger.info("Bar plot сохранен.")
    
    # Beeswarm для одного класса
    target_class = 'supermarkety'
    
    if target_class in config.CATS_TARGET:
        idx = config.CATS_TARGET.index(target_class)
        logger.info(f"Строим Beeswarm для: {target_class} (index {idx})")
        
        shap_matrix_class = shap_values_np[:, :, idx]
        
        logger.info(f"Class matrix shape: {shap_matrix_class.shape}")
        
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_matrix_class, 
            X_explain_df, 
            show=False,
            max_display=20
        )
        plt.title(f"Feature Impact on '{target_class}'")
        plt.tight_layout()
        plt.savefig(f"{config.ARTIFACTS_DIR}/shap_beeswarm_{target_class}.png")
        plt.close()
        logger.info(f"Beeswarm plot сохранен.")
    else:
        logger.warning(f"Класс {target_class} не найден.")


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()
    