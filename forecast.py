import numpy as np
import torch
import torch.cuda.amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser() 
parser.add_argument('Data', help='')
args = parser.parse_args()


import logging
logger = logging.getLogger(f'{args.Data}')
logging.basicConfig(level=logging.INFO,
                    filename=f'{args.Data}',
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )

from momentfm.utils.utils import control_randomness
from momentfm.data.informer_dataset import InformerDataset
from momentfm.utils.forecasting_metrics import get_forecasting_metrics
from momentfm import L2pMOMENTPipeline
model = L2pMOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large", 
    model_kwargs={
        'task_name': 'forecasting',
        'forecast_horizon': 192,
        'head_dropout': 0.1,
        'weight_decay': 0,
        'freeze_encoder': True, # Freeze the patch embedding layer
        'freeze_embedder': True, # Freeze the transformer encoder
        'freeze_head': False, # The linear forecasting head must be trained
        
        'prompt_pool': True,
        'use_prompt_mask': True,
        'prompt_length': 16,           
        'embedding_key': 'mean',
        'prompt_key': True,
        'prompt_key_init': 'zero',
        'pool_size': 8,
        'top_k': 4,
        'batchwise_prompt': None
    },
)
model.init()
# Set random seeds for PyTorch, Numpy etc.
control_randomness(seed=13) 

BatachSize=64
NumWorkers=8
# Load data
train_dataset = InformerDataset(data_split="train", random_seed=13, forecast_horizon=192)
train_loader = DataLoader(train_dataset, batch_size=BatachSize, num_workers=NumWorkers,shuffle=True,drop_last=True)

val_dataset = InformerDataset(data_split="val", random_seed=13, forecast_horizon=192)
val_loader = DataLoader(val_dataset, batch_size=BatachSize, num_workers=NumWorkers,shuffle=True,drop_last=True)

test_dataset = InformerDataset(data_split="test", random_seed=13, forecast_horizon=192)
test_loader = DataLoader(test_dataset, batch_size=BatachSize, num_workers=NumWorkers,shuffle=True,drop_last=True)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cur_epoch = 0
max_epoch = 10

# Move the model to the GPU
model = model.to(device)

# Move the loss function to the GPU
criterion = criterion.to(device)

# Enable mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Create a OneCycleLR scheduler
max_lr = 1e-4
total_steps = len(train_loader) * max_epoch
scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3)

# Gradient clipping value
max_norm = 5.0

while cur_epoch < max_epoch:
    model.train()
    losses = []
    for timeseries, forecast, input_mask in tqdm(train_loader, total=len(train_loader)):
        # Move the data to the GPU
        timeseries = timeseries.float().to(device)
        input_mask = input_mask.to(device)
        forecast = forecast.float().to(device)
        # print(f'input_mask: {input_mask.shape}')
        with torch.cuda.amp.autocast():
            output = model(timeseries, input_mask)
        
        loss = criterion(output.forecast, forecast)

        # Scales the loss for mixed precision training
        scaler.scale(loss).backward()

        # Clip gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item())

    losses = np.array(losses)
    average_loss = np.average(losses)
    logger.info(f"Epoch {cur_epoch}: Train loss: {average_loss:.3f}")
    
    # Step the learning rate scheduler
    scheduler.step()
    
    
    # Evaluate the model on the test split

    logger.info('===============Evalutate===============')
    trues, preds, histories, losses = [], [], [], []
    model.eval()
    with torch.no_grad():
        for timeseries, forecast, input_mask in tqdm(val_loader, total=len(val_loader)):
        # Move the data to the GPU
            timeseries = timeseries.float().to(device)
            input_mask = input_mask.to(device)
            forecast = forecast.float().to(device)

            with torch.cuda.amp.autocast():
                output = model(timeseries, input_mask)
            
            loss = criterion(output.forecast, forecast)                
            losses.append(loss.item())

            trues.append(forecast.detach().cpu().numpy())
            preds.append(output.forecast.detach().cpu().numpy())
            histories.append(timeseries.detach().cpu().numpy())
    
    losses = np.array(losses)
    average_loss = np.average(losses)

    trues = np.concatenate(trues, axis=0)
    preds = np.concatenate(preds, axis=0)
    histories = np.concatenate(histories, axis=0)
    
    metrics = get_forecasting_metrics(y=trues, y_hat=preds, reduction='mean')

    logger.info(f"Epoch {cur_epoch}: Val MSE: {metrics.mse:.3f} | Val MAE: {metrics.mae:.3f}")
    if cur_epoch==0 or cur_epoch==9:
        logger.info('===============Test===============')
        trues, preds, histories, losses = [], [], [], []
        model.eval()
        with torch.no_grad():
            for timeseries, forecast, input_mask in tqdm(test_loader, total=len(test_loader)):
            # Move the data to the GPU
                timeseries = timeseries.float().to(device)
                input_mask = input_mask.to(device)
                forecast = forecast.float().to(device)

                with torch.cuda.amp.autocast():
                    output = model(timeseries, input_mask)
                
                loss = criterion(output.forecast, forecast)                
                losses.append(loss.item())

                trues.append(forecast.detach().cpu().numpy())
                preds.append(output.forecast.detach().cpu().numpy())
                histories.append(timeseries.detach().cpu().numpy())


        trues = np.concatenate(trues, axis=0)
        preds = np.concatenate(preds, axis=0)
        histories = np.concatenate(histories, axis=0)

        metrics = get_forecasting_metrics(y=trues, y_hat=preds, reduction='mean')

        logger.info(f"Epoch {cur_epoch}: Test MSE: {metrics.mse:.3f} | Test MAE: {metrics.mae:.3f}")
    cur_epoch += 1
