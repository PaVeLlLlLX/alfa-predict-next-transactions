import sys
import os
from typing import List
sys.path.append(os.getcwd())

import torch
import pandas as pd
import numpy as np
import joblib
import json
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
from src import config, dataset, model, utils
from torch.amp import autocast

logger = utils.get_logger()
device_str = "cuda" if torch.cuda.is_available() else "cpu"

def main() -> None:
    utils.seed_everything(config.SEED)
    
    logger.info("Загрузка ресурсов")
    
    if os.path.exists(config.THRESHOLDS_PATH):
        with open(config.THRESHOLDS_PATH, 'r') as f:
            thresholds = np.array(json.load(f))
        logger.info("Загружены оптимизированные пороги.")
    else:
        logger.warning("Файл порогов не найден. Используются стандартные 0.5.")
        thresholds = np.full(32, 0.5)

    svd = joblib.load(config.SVD_PATH)
    mcc_map = pd.read_excel(config.MCC_MAP_PATH)
    mcc_dict = dict(zip(mcc_map["mcc"].astype(str), mcc_map["eng_cat"]))
    
    with open(config.VOCAB_PATH, 'r') as f: vocabs = json.load(f)
    encoder = dataset.CategoricalEncoder(vocabs)
    processor = dataset.DataProcessor(svd, mcc_dict)
    cards = [len(vocabs[c]) for c in config.CAT_COLS]
    
    ftt = model.MultiLabelFTTransformer(len(config.NUM_COLS), cards).to(config.DEVICE)
    ftt.load_state_dict(torch.load(config.MODEL_CHECKPOINT, map_location=config.DEVICE))
    ftt.eval()
    
    logger.info("Инференс на тесте")
    test_df = pd.read_csv(config.TEST_SOURCE)
    test_df['target'] = 0
    temp_file = "temp_test.parquet"
    test_df.to_parquet(temp_file, index=False)
    
    test_ds = dataset.TabularStreamingDataset([temp_file], encoder, processor, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE*2, num_workers=2)
    
    test_preds_list: List[np.ndarray] = []
    test_len = math.ceil(len(test_df) / (config.BATCH_SIZE*2))
    
    with torch.no_grad():
        for batch in tqdm(test_loader, total=test_len, desc="Predicting"):
            with autocast(device_str):
                logits = ftt(batch['x_num'].to(config.DEVICE), batch['x_cat'].to(config.DEVICE))
            test_preds_list.append(torch.sigmoid(logits).cpu().numpy())
            
    y_test_prob = np.vstack(test_preds_list)
    if os.path.exists(temp_file): os.remove(temp_file)

    logger.info("Применение порогов")
    
    y_test_bin = np.zeros_like(y_test_prob, dtype=int)
    for i in range(32):
        y_test_bin[:, i] = (y_test_prob[:, i] >= thresholds[i]).astype(int)
    
    rows_no_positive = np.where(y_test_bin.sum(axis=1) == 0)[0]
    if len(rows_no_positive):
        logger.info(f"Исправлено пустых строк: {len(rows_no_positive)}")
        argmax_idx = y_test_prob[rows_no_positive].argmax(axis=1)
        y_test_bin[rows_no_positive, argmax_idx] = 1
        
    submission = pd.DataFrame(y_test_bin, columns=config.CATS_TARGET)
    submission.insert(0, 'id', test_df.index)
    
    sub_name = f"submission.csv"
    submission.to_csv(sub_name, index=False)
    logger.info(f"Сабмит готов: {sub_name}")

if __name__ == "__main__":
    main()