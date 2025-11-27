import sys
import os
from typing import List, Tuple
sys.path.append(os.getcwd())

import torch
import numpy as np
import pandas as pd
import joblib
import json
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
import pyarrow.parquet as pq
from src import config, dataset, model, utils
from torch.amp import autocast

logger = utils.get_logger()
device_str = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    utils.seed_everything(config.SEED)
    
    logger.info("Загрузка ресурсов")
    svd = joblib.load(config.SVD_PATH)
    mcc_map = pd.read_excel(config.MCC_MAP_PATH)
    mcc_dict = dict(zip(mcc_map["mcc"].astype(str), mcc_map["eng_cat"]))
    
    with open(config.VOCAB_PATH, 'r') as f: vocabs = json.load(f)
    encoder = dataset.CategoricalEncoder(vocabs)
    processor = dataset.DataProcessor(svd, mcc_dict)
    cards: List[int] = [len(vocabs[c]) for c in config.CAT_COLS]
    
    ftt = model.MultiLabelFTTransformer(len(config.NUM_COLS), cards).to(config.DEVICE)
    ftt.load_state_dict(torch.load(config.MODEL_CHECKPOINT, map_location=config.DEVICE))
    ftt.eval()
    
    logger.info("Генерация вероятностей на валидации")
    val_ds = dataset.TabularStreamingDataset([config.VAL_FINAL], encoder, processor)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE*2, num_workers=4)
    
    val_preds_list: List[np.ndarray] = []
    val_true_list: List[np.ndarray] = []
    
    val_len = math.ceil(pq.read_metadata(config.VAL_FINAL).num_rows / (config.BATCH_SIZE*2))
    
    with torch.no_grad():
        for batch in tqdm(val_loader, total=val_len, desc="Validating"):
            with autocast(device_str):
                logits = ftt(batch['x_num'].to(config.DEVICE), batch['x_cat'].to(config.DEVICE))
            val_preds_list.append(torch.sigmoid(logits).cpu().numpy())
            val_true_list.append(batch['labels'].numpy())
            
    y_val_prob = np.vstack(val_preds_list)
    y_val_true = np.vstack(val_true_list).astype(int)
    
    logger.info("Запуск поиска порогов")
    
    current_thresholds = np.full(32, 0.5)
    current_binary_matrix = (y_val_prob >= current_thresholds).astype(int)
    
    best_global_score = utils.hamming_score_weighted(y_val_true, current_binary_matrix, config.WEIGHTS)
    logger.info(f"Стартовый скор (все 0.5): {best_global_score:.5f}")
    
    grid = np.arange(0.3, 0.96, 0.04)
    CYCLES = 2
    
    for cycle in range(CYCLES):
        logger.info(f"--- Cycle {cycle + 1}/{CYCLES} ---")
        
        for col_idx in tqdm(range(32), desc="Optimizing columns"):
            best_t_for_col = current_thresholds[col_idx]
            best_score_iter = best_global_score
            
            for t in grid:
                current_binary_matrix[:, col_idx] = (y_val_prob[:, col_idx] >= t).astype(int)
                score = utils.hamming_score_weighted(y_val_true, current_binary_matrix, config.WEIGHTS)
                
                if score > best_score_iter:
                    best_score_iter = score
                    best_t_for_col = t
            
            current_thresholds[col_idx] = best_t_for_col
            current_binary_matrix[:, col_idx] = (y_val_prob[:, col_idx] >= best_t_for_col).astype(int)
            best_global_score = best_score_iter
            
        logger.info(f"Скор после цикла {cycle + 1}: {best_global_score:.5f}")
    logger.info(f"\nИтоговые пороги: {current_thresholds}")

    logger.info(f"Итоговые пороги сохранены в: {config.THRESHOLDS_PATH}")
    with open(config.THRESHOLDS_PATH, 'w') as f:
        json.dump(current_thresholds.tolist(), f)

if __name__ == "__main__":
    main()