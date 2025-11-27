import sys
import os
sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader
from src import config, dataset, model, utils
import joblib
import pandas as pd
import json
import math
from tqdm import tqdm
import pyarrow.parquet as pq
from torch.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup


logger = utils.get_logger()

def main():
    utils.seed_everything(config.SEED)
    
    logger.info("Загрузка ресурсов")
    svd = joblib.load(config.SVD_PATH)
    mcc_map = pd.read_excel(config.MCC_MAP_PATH)
    mcc_dict = dict(zip(mcc_map["mcc"].astype(str), mcc_map["eng_cat"]))
    
    with open(config.VOCAB_PATH, 'r') as f: vocabs = json.load(f)
    encoder = dataset.CategoricalEncoder(vocabs)
    processor = dataset.DataProcessor(svd, mcc_dict)
    
    cards = [len(vocabs[c]) for c in config.CAT_COLS]
    logger.info(f"Cards: {cards}")
    
    train_ds = dataset.TabularStreamingDataset([config.TRAIN_FINAL], encoder, processor)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, num_workers=4, pin_memory=True)
    
    logger.info("Инициализация модели")
    ftt = model.MultiLabelFTTransformer(len(config.NUM_COLS), cards).to(config.DEVICE)
    
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optim_groups = [
        {"params": [p for n, p in ftt.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": config.WD},
        {"params": [p for n, p in ftt.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=config.LR)
    loss_fn = model.AsymmetricLoss()
    scaler = GradScaler()
    
    # Scheduler
    train_rows = pq.read_metadata(config.TRAIN_FINAL).num_rows
    steps_per_epoch = math.ceil(train_rows / config.BATCH_SIZE)
    total_steps = steps_per_epoch * config.EPOCHS
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )
    
    logger.info("Старт обучения на {config.EPOCHS} эпох")
    for epoch in range(config.EPOCHS):
        ftt.train()
        running_loss = 0
        pbar = tqdm(train_loader, total=steps_per_epoch, desc=f"Epoch {epoch+1}")
        
        for i, batch in enumerate(pbar):
            x_num, x_cat, y = batch['x_num'].to(config.DEVICE), batch['x_cat'].to(config.DEVICE), batch['labels'].to(config.DEVICE)
            
            with autocast("cuda"):
                loss = loss_fn(ftt(x_num, x_cat), y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            if i % 500 == 0 and i > 0:
                avg_loss = running_loss / 500
                pbar.set_postfix({'loss': avg_loss})
                if i % 2000 == 0:
                    logger.info(f"Epoch {epoch+1}, Step {i}, Loss: {avg_loss:.4f}")
                running_loss = 0
        
        logger.info(f"Эпоха {epoch+1} завершена. Сохранение чекпоинта.")
        torch.save(ftt.state_dict(), f"{config.ARTIFACTS_DIR}/ftt_epoch_{epoch+1}.pt")
        #torch.save(ftt.state_dict(), config.MODEL_CHECKPOINT)

if __name__ == "__main__":
    main()