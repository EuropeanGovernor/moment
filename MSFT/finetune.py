import math
import numpy as np

import torch
from torch import nn
import torch.cuda.amp
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.distributions import Distribution
from torch.utils.tensorboard import SummaryWriter
from head import ForecastingHead
from transformers import T5Config
from attention import T5EncoderModel_LoRA, T5EncoderModel
from EarlyStop import EarlyStopping

import os
import sys
import itertools
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname((os.getcwd())))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from momentfm.utils.utils import control_randomness
from momentfm.models.layers.revin import RevIN
from momentfm.utils.masking import Masking
from momentfm.data.informer_dataset import InformerDataset 
from momentfm.utils.forecasting_metrics import get_forecasting_metrics
from momentfm.models.layers.embed import PatchEmbedding
control_randomness(seed=13) 

class MomentFinetune():
    def __init__(self, args):
        super().__init__()
        self.dataset = args.dataset
        self.train_bs = args.train_bs
        self.eval_bs = args.eval_bs
        self.max_epoch = args.max_epoch
        self.max_norm = 5.0
        self.NumWorkers = 4
        self.BestEvalLoss = float("inf")
        self.patience = args.patience
        self.criterion = nn.MSELoss()
        self.scaler = GradScaler() 
        self.init_lr = args.init_lr
        self.head_lr = args.head_lr
        self.scale_weight_lr = args.scale_weight_lr
        self.weight_decay = args.weight_decay
        self.device_name = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name = args.version
        self.d_model = {"small":512, "base":768, "large":1024}[self.model_name]
        self.seq_len = 512
        self.patch_size = 8
        self.ds_factor = 2 
        self.pred_length = args.pred_length
        self.patch_stride_len = 8
        self.scale_num_patch = {}
        self.mask_generator = Masking()
        self.early_stopping = EarlyStopping(
            patience=self.patience, 
            verbose=True,
            delta=0.0001,
            path=f"{self.dataset}_{self.pred_length}_best_model.pth",
            trace_func=print
        )
        # model out后找到mask对应的位置
        self.SCALE_INDEX = {96:[
                list(range(64, 76)),   # Scale 0: 从索引 64 到 75（12 个 Patch） 
                list(range(108, 114)), # Scale 1: 从索引 108 到 113（6 个 Patch）
                list(range(130, 133)), # Scale 2: 从索引 130 到 132（3 个 Patch）
                list(range(141, 143))  # Scale 3: 从索引 141 到 142（2 个 Patch）
                ],
                            192:[
                list(range(64, 88)),
                list(range(120, 132)),
                list(range(148, 154)),
                list(range(162, 165))
                ],
                            336: [
                list(range(64, 112)), 
                list(range(128, 152)), 
                list(range(160, 172)),
                list(range(180, 186))  
                ],
                            720: [
                list(range(64, 160)),
                list(range(180, 228)),
                list(range(240, 264)),
                list(range(280, 292)) 
            ]}[self.pred_length]
        
        # 不同pred length对应的mask所占patch长度
        self.NUM_PATCH = {96:[12, 6, 3, 2],
                          192:[24, 12, 6, 3],
                          192:[48, 24, 12, 6],
                          192:[96, 48, 24, 12]
                          }[self.pred_length]

        # 新模块参数
        self.lora = args.lora
        self.linr = args.linear
        self.head_dropout = args.head_dropout
        self.head_nf = [self.d_model * _ for _ in self.NUM_PATCH]
        self.scale_weights = nn.Parameter(torch.ones(4, device=self.device_name)) 
        self.linear = nn.ModuleList([nn.Linear(self.d_model, self.d_model) for _ in range(4)]).to(self.device_name) 

        # 记录scale_loss的列表
        self.scale_loss_history = [[] for _ in range(4)]
        self.L = []
        
        # 初始化预测头
        pred_length = 2*self.pred_length  # 2*初始预测长度
        self.heads = nn.ModuleList([
            ForecastingHead(_, pred_length, self.head_dropout).to(self.device_name)
            for _ in self.head_nf
            if (pred_length := max(1, pred_length // 2))  # 确保最小值不小于1
        ])

    def _build_model(self):
        config = T5Config(
                            architectures=["T5ForConditionalGeneration"],
                            classifier_dropout=0.0,
                            d_ff={"small":1024, "base":2048, "large":2816}[self.model_name],
                            d_kv=64,
                            d_model={"small":512, "base":768, "large":1024}[self.model_name],
                            decoder_start_token_id=0,
                            dense_act_fn="gelu_new",
                            dropout_rate=0.1,
                            eos_token_id=1,
                            feed_forward_proj="gated-gelu",
                            initializer_factor=1.0,
                            is_encoder_decoder=True,
                            is_gated_act=True,
                            layer_norm_epsilon=1e-06,
                            model_type="t5",
                            n_positions=512,
                            num_decoder_layers={"small":8, "base":12, "large":24}[self.model_name],
                            num_heads={"small":6, "base":12, "large":16}[self.model_name],
                            num_layers={"small":8, "base":12, "large":24}[self.model_name],
                            output_past=True,
                            pad_token_id=0,
                            relative_attention_max_distance=128,
                            relative_attention_num_buckets=32,
                            tie_word_embeddings=False,
                            transformers_version="4.33.3",
                            use_cache=True,
                            vocab_size=32128
                        )
        self.normalizer = RevIN(
            num_features=1, affine=False
        )
        print(config)
        if self.lora: self.model =  T5EncoderModel_LoRA(config,pred_length=self.pred_length).get_encoder()
        else: self.model =  T5EncoderModel(config).get_encoder()

        checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),f"MOMENT_{self.model_name}_{self.pred_length}.ckpt")
        self.checkpoint = torch.load(checkpoint_path, map_location="cpu")

        encoder_state_dict = {k.replace("encoder.", ""): v for k, v in self.checkpoint.items() if k.startswith("encoder.")}
        missing_keys, unexpected_keys = self.model.load_state_dict(encoder_state_dict, strict=False)

        self._get_embeder()

        print(f"✅ 成功加载参数: {encoder_state_dict.keys()-missing_keys}")
        print(f"❌ Checkpoint中没有的参数: {missing_keys}")
        print(f"⚠️ 模型中没有的参数: {unexpected_keys}")

        self.model.to(self.device_name)

        # 冻结训练参数
        for name, param in self.model.named_parameters():
            # 冻结所有 self-attention 层的 base_layer
            if "SelfAttention.q" in name or "SelfAttention.k" in name or "SelfAttention.v" in name or "DenseReluDense" in name:
                param.requires_grad = False

        print(self.model)

        # 查看需训练参数
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

        for name, param in enumerate(self.patch_embedding.parameters()):
            if param.requires_grad:
                print(name, param.shape)

        for i, param in enumerate(self.linear.parameters()):
            if param.requires_grad:
                print(f"linear_{i}", param.shape)

        for i, param in enumerate(self.heads.parameters()):
            if param.requires_grad:
                print(f"heads_{i}", param.shape)

        
    def _get_lr_schedular(self):
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr= [self.init_lr , self.head_lr , self.scale_weight_lr],
                epochs=self.max_epoch,
                steps_per_epoch=len(self.dataloader["train"]),
            )
    def _get_optimizer(self):
        self.optimizer = optim.AdamW(
            [
                {
                    'params': itertools.chain(self.model.parameters(), 
                                self.patch_embedding.parameters(),
                                self.linear.parameters()),
                    'lr': self.init_lr, 
                    'weight_decay': self.weight_decay
                },
                {
                    'params':  self.heads.parameters(),
                    'lr': self.head_lr,
                    'weight_decay': self.weight_decay
                },
                {
                    'params': self.scale_weights,
                    'lr': self.scale_weight_lr,
                    'weight_decay': self.weight_decay
                }
            ]
        )

    def _get_dataloader(self):
        train_dataset = InformerDataset(data_split="train", random_seed=13, forecast_horizon=self.pred_length, full_file_path_and_name=f"./long_term_forecast/ETT-small/{self.dataset}.csv")
        val_dataset = InformerDataset(data_split="val", random_seed=13, forecast_horizon=self.pred_length, full_file_path_and_name=f"./long_term_forecast/ETT-small/{self.dataset}.csv")
        test_dataset = InformerDataset(data_split="test", random_seed=13, forecast_horizon=self.pred_length, full_file_path_and_name=f"./long_term_forecast/ETT-small/{self.dataset}.csv")

        train_loader = DataLoader(train_dataset, batch_size=self.train_bs, num_workers=self.NumWorkers, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.eval_bs, num_workers=self.NumWorkers, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.eval_bs, num_workers=self.NumWorkers, shuffle=True, pin_memory=True)
        self.dataloader = {"train":train_loader, "val":val_loader, "test":test_loader}

    def _downsample(self, timeseries, forecast, input_mask) -> tuple:
        """
        对时间序列和预测序列进行三次降采样
        
        参数:
        timeseries: [batch_size, channel, context_length]
        forecast: [batch_size, channel, predict_length]
        
        返回:
        scale_ts: 在num_patch维度拼接的timeseries+forecast, 共四个尺度
        scale_seq_mask: 在一个sequence中表示哪些是padding过的[batch_size, sequence_length]
        scale_patch_mask: 在一个patch sequence中表示哪些是需要预测的[batch_size, num_patch]
        """
        batch_size, channel, context_length = timeseries.shape
        _, _, prediction_length = forecast.shape
        scale_ts, scale_fc, scale_seq_mask, scale_patch_mask = [], [], [], []

        timeseries = self.normalizer(x=timeseries, mask=input_mask, mode="norm")
        timeseries = torch.nan_to_num(timeseries, nan=0, posinf=0, neginf=0)

        new_context_length = context_length
        new_prediction_length = prediction_length
        new_timeseries = timeseries
        new_forecast = forecast
        padding_needed = 0

        for i in range(4):
            self.ds_factor = 1 if i == 0 else 2
            new_context_length = math.ceil(new_context_length / self.ds_factor)
            new_prediction_length = math.ceil(new_prediction_length / self.ds_factor)

            new_timeseries = nn.functional.avg_pool1d(new_timeseries, kernel_size=self.ds_factor, stride=self.ds_factor)
            new_forecast = nn.functional.avg_pool1d(new_forecast, kernel_size=self.ds_factor, stride=self.ds_factor)
            scale_fc.append(new_forecast)
            
            if new_prediction_length % self.patch_size !=0:
                padding_needed = self.patch_size - (new_prediction_length % self.patch_size)
                new_forecast = torch.nn.functional.pad(new_forecast, (0,padding_needed))
            
            # 拼接context和预测部分
            ts_patch = new_timeseries.reshape(batch_size, channel, -1, self.patch_size)          
            fc_patch = new_forecast.reshape(batch_size, channel, -1, self.patch_size)
            combined = torch.cat([ts_patch, fc_patch], dim=2).to(torch.float32)
            scale_ts.append(combined)

            # 生成scale_seq_mask
            ones = torch.ones(batch_size, new_context_length + new_prediction_length) 
            zeros = torch.zeros(batch_size, padding_needed)  
            seq_mask = torch.cat([ones, zeros], dim=1)
            scale_seq_mask.append(seq_mask)

            # 生成scale_patch_mask
            ones = torch.ones(batch_size, ts_patch.shape[2]) 
            zeros = torch.zeros(batch_size, fc_patch.shape[2])  
            patch_mask = torch.cat([ones, zeros], dim=1)
            scale_patch_mask.append(patch_mask)

            # 每个scale占多少个patch
            self.scale_num_patch[i] = combined.shape[2]
        return scale_ts, scale_fc, scale_seq_mask, scale_patch_mask
        
    def _upsample(self,  input_tensor) -> tuple:
        batch_size, n_channel, length = input_tensor.shape
        
        factor = self.pred_length // length
        upsampled_tensor = input_tensor.repeat_interleave(factor, dim=2)
        return upsampled_tensor

    def _get_embeder(self):
        self.patch_embedding = PatchEmbedding(
            d_model=self.d_model,
            seq_len=self.seq_len,
            patch_len=self.patch_size,
            stride=self.patch_stride_len,
            patch_dropout= 0.1,
            add_positional_embedding= True,
            value_embedding_bias= False,
            orth_gain=1.41,
        )

        # 重新映射 checkpoint 里的 key,加载到PatchEmbedding中
        patch_embedding_checkpoint = {key.replace("patch_embedding.", ""): value for key, value in self.checkpoint.items()}
        self.patch_embedding.load_state_dict(patch_embedding_checkpoint, strict=False)    
        self.patch_embedding = self.freeze_parameters(self.patch_embedding)
    def freeze_parameters(self, model):
        """
        Freeze parameters of the model
        ❄️:patch_embedding/encoder
        """
        # Freeze the parameters
        for name, param in model.named_parameters():
            param.requires_grad = False

        return model
    
    def embed(self, x, input_masks):
        input_embed = [self.patch_embedding(_, input_mask).to(self.device_name) for _, input_mask in zip(x, input_masks)]
        return input_embed        
    
    def cal_scale_loss(self, scale_ts, scale_fc, step):
        """
        计算不同尺度的损失并加权求和。
        
        参数:
            scale_ts: 预测头得到的不同scale预测结果组成的列表
            scale_fc: 降采样得到的不同scale真值组成的列表
            step: 当前训练步骤
        """
        L = 0.0  # 初始化总损失
        scale_weights = torch.softmax(self.scale_weights.float(), dim=0)  # 确保权重为 float32

        for i, (ts, fc) in enumerate(zip(scale_ts, scale_fc)):
            ts, fc = ts.to(self.device_name, dtype=torch.float16), fc.to(self.device_name, dtype=torch.float16)
            scale_loss = self.criterion(ts, fc)
            L += scale_loss * scale_weights[i]
    
            # 将当前尺度的损失记录到历史中
            self.scale_loss_history[i].append(scale_loss.item())

            # 每 100 个步骤记录一次每个尺度的损失
            if step % 100 == 0:
                scale_avg_loss = np.mean(self.scale_loss_history[i][-100:])  # 计算最近 100 个步骤的平均损失
                writer.add_scalar(f"Loss/scale_{i}", scale_avg_loss, global_step=step)

        self.L.append(L.item())
        # 每 100 个步骤记录一次加权总损失
        if step % 100 == 0:
            sum_scale_avg_loss = np.mean(self.L[-100:])  # 计算最近 100 个步骤的平均损失
            writer.add_scalar(f"Loss/scale_sum", sum_scale_avg_loss, global_step=step)

        return L.to(torch.float16).to(self.device_name)  # 确保返回值类型正确

    def train(self):
        step = 0 
        for epoch in range(self.max_epoch):
            self.model.train()
            self.patch_embedding.train()
            self.linear.train()
            self.heads.train()
            losses = []
            for timeseries, forecast, input_mask in tqdm(self.dataloader["train"], total=len(self.dataloader["train"])):
                step += 1
                
                # 获取batch_size, n_channels
                batch_size, n_channels, _ = timeseries.shape

                # 降采样得到数据和scale_ts/input_seq_mask/input_patch_mask (4 scale)
                scale_ts, scale_fc, input_seq_mask, input_patch_mask = self._downsample(timeseries, forecast, input_mask)
                n_patches = sum([ _.shape[2] for _ in scale_ts])

                # 先Embedding（含Mask Embedding），再投影
                input_embed = self.embed(scale_ts, input_patch_mask)
                # input_embed = self.embed(scale_ts, [torch.ones_like(_) for _ in input_patch_mask])

                # 分不同尺度投影
                if self.linr: input_embed = [linear(inp) for linear, inp in zip(self.linear, input_embed)]

                # 先Embedding（不含Mask Embedding），然后投影，再Mask Embedding
                # input_embed = [self.patch_embedding.apply_mask_embedding(_, input_mask).to(self.device_name) for _, input_mask in zip(input_embed, input_patch_mask)]

                # 进入encoder之前拼接不同尺度
                enc_in = torch.cat([_ for _ in input_embed], dim=2)
                enc_in = enc_in.reshape(batch_size*n_channels, n_patches,self.d_model).to(self.device_name)

                attn_mask = torch.cat([_ for _ in input_patch_mask], dim=1)
                attn_mask = attn_mask.unsqueeze(1).repeat(1, n_channels, 1)
                attn_mask = attn_mask.reshape(batch_size*n_channels, n_patches).to(self.device_name)
                
                with torch.amp.autocast(device_type='cuda',  dtype=torch.float16):
                    output = self.model(inputs_embeds = enc_in, attention_mask = attn_mask)
                
                    enc_out = output.last_hidden_state
                    enc_out = enc_out.reshape((-1, n_channels, n_patches, self.d_model))
                    mask_out = [enc_out[..., indices, :] for indices in self.SCALE_INDEX] 
                    head_out = [head(mask) for head, mask in zip(self.heads,mask_out)]
                    denorm_out = [self.normalizer(x=_.to(self.device_name), mode="denorm") for _ in head_out]

                    loss = self.cal_scale_loss(denorm_out, scale_fc, step)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                torch.nn.utils.clip_grad_norm_(self.linear.parameters(), self.max_norm)
                torch.nn.utils.clip_grad_norm_(self.heads.parameters(), self.max_norm)
                torch.nn.utils.clip_grad_norm_(self.scale_weights, self.max_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                losses.append(loss.item())

                # 每 100 个步骤记录一次训练损失
                if step % 100 == 0:
                    avg_loss = np.mean(losses[-100:])  # 计算最近 100 个步骤的平均损失
                    writer.add_scalar("Loss/train_step", avg_loss, step)
                    print(f"Epoch [{epoch+1}/{self.max_epoch}] Step [{step}] - Train Loss: {avg_loss:.3f}")

            losses = np.array(losses)
            average_loss = np.average(losses)
            writer.add_scalar("Loss/train", average_loss, global_step=epoch)

            scale_weights = torch.softmax(self.scale_weights, dim=0)
            for num in range(scale_weights.shape[0]):
                writer.add_scalar(f'scale_weights/{num}', scale_weights[num], global_step=epoch)

            self.lr_scheduler.step()

            early_stop = self.eval(cur_epoch=epoch)
            if early_stop:
                print(f"Training stopped early at epoch {epoch+1}/{self.max_epoch}")
                break

    def eval(self, cur_epoch):
        trues, preds, histories, losses = [], [], [], []
        self.model.eval()
        self.patch_embedding.eval()
        self.linear.eval()
        self.heads.eval()
        
        with torch.no_grad():
            for timeseries, forecast, input_mask in tqdm(self.dataloader["val"], total=len(self.dataloader["val"])):
                # 获取batch_size, n_channels
                batch_size, n_channels, _ = timeseries.shape

                # 降采样得到数据和scale_ts/input_seq_mask/input_patch_mask (4 scale)
                scale_ts, scale_fc, input_seq_mask, input_patch_mask = self._downsample(timeseries, forecast, input_mask)
                n_patches = sum([ _.shape[2] for _ in scale_ts])

                # 先Embedding（含Mask Embedding），再投影
                input_embed = self.embed(scale_ts, input_patch_mask)
                # input_embed = self.embed(scale_ts, [torch.ones_like(_) for _ in input_patch_mask])


                # 分不同尺度投影
                if self.linr: input_embed = [linear(inp) for linear, inp in zip(self.linear, input_embed)]

                # 先Embedding（不含Mask Embedding），然后投影，再Mask Embedding
                # input_embed = [self.patch_embedding.apply_mask_embedding(_, input_mask).to(self.device_name) for _, input_mask in zip(input_embed, input_patch_mask)]

                # 进入encoder之前拼接不同尺度
                enc_in = torch.cat([_ for _ in input_embed], dim=2)
                enc_in = enc_in.reshape(batch_size*n_channels, n_patches,self.d_model).to(self.device_name)

                attn_mask = torch.cat([_ for _ in input_patch_mask], dim=1)
                attn_mask = attn_mask.unsqueeze(1).repeat(1, n_channels, 1)
                attn_mask = attn_mask.reshape(batch_size*n_channels, n_patches).to(self.device_name)

                with torch.amp.autocast(device_type='cuda'):
                    output = self.model(inputs_embeds = enc_in, attention_mask = attn_mask.to(torch.bool))
                
                enc_out = output.last_hidden_state
                enc_out = enc_out.reshape((-1, n_channels, n_patches, self.d_model))
                mask_out = [enc_out[..., indices, :] for indices in self.SCALE_INDEX] 
                head_out = [head(mask) for head, mask in zip(self.heads,mask_out)]
                denorm_out = [self.normalizer(x=_, mode="denorm") for _ in head_out]

                # upsampling
                up_sampling = [self._upsample(head) for head in denorm_out]

                weighted_forecast = 0
                scale_weights = torch.softmax(self.scale_weights, dim=0)
                for i, weight in enumerate(scale_weights):
                    weighted_forecast += weight * up_sampling[i] 
                
                loss = self.criterion(weighted_forecast.float(), forecast.float().to(self.device_name))
                losses.append(loss.item())
                trues.append(forecast.detach().cpu().numpy())
                preds.append(weighted_forecast.detach().cpu().numpy())
                histories.append(timeseries.detach().cpu().numpy())

            losses = np.array(losses)
            average_loss = np.average(losses)
            trues = np.concatenate(trues, axis=0)
            preds = np.concatenate(preds, axis=0)
            histories = np.concatenate(histories, axis=0)
            
            metrics = get_forecasting_metrics(y=trues, y_hat=preds, reduction="mean")

            writer.add_scalar(f"Loss/Val", average_loss, global_step=cur_epoch)
            writer.add_scalar(f"Val/MSE", metrics.mse, global_step=cur_epoch)
            writer.add_scalar(f"Val/MAE", metrics.mae, global_step=cur_epoch)

            if self.BestEvalLoss > average_loss:
                self.BestEvalLoss = average_loss
                self.test(cur_epoch)
                self.save_model()
             
            self.early_stopping(average_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                return True  
            
            return False 
    def test(self, cur_epoch):
        trues, preds, histories, losses = [], [], [], []
        self.model.eval()
        self.patch_embedding.eval()
        self.linear.eval()
        self.heads.eval()
        
        with torch.no_grad():
            for timeseries, forecast, input_mask in tqdm(self.dataloader["test"], total=len(self.dataloader["test"])):
                # 获取batch_size, n_channels
                batch_size, n_channels, _ = timeseries.shape

                # 降采样得到数据和scale_ts/input_seq_mask/input_patch_mask (4 scale)
                scale_ts, scale_fc, input_seq_mask, input_patch_mask = self._downsample(timeseries, forecast, input_mask)
                n_patches = sum([ _.shape[2] for _ in scale_ts])

                # 先Embedding（含Mask Embedding），再投影
                input_embed = self.embed(scale_ts, input_patch_mask)
                # input_embed = self.embed(scale_ts, [torch.ones_like(_) for _ in input_patch_mask])

                # 分不同尺度投影
                if self.linr: input_embed = [linear(inp) for linear, inp in zip(self.linear, input_embed)]

                # 先Embedding（不含Mask Embedding），然后投影，再Mask Embedding
                # input_embed = [self.patch_embedding.apply_mask_embedding(_, input_mask).to(self.device_name) for _, input_mask in zip(input_embed, input_patch_mask)]

                # 进入encoder之前拼接不同尺度
                enc_in = torch.cat([_ for _ in input_embed], dim=2)
                enc_in = enc_in.reshape(batch_size*n_channels, n_patches,self.d_model).to(self.device_name)

                attn_mask = torch.cat([_ for _ in input_patch_mask], dim=1)
                attn_mask = attn_mask.unsqueeze(1).repeat(1, n_channels, 1)
                attn_mask = attn_mask.reshape(batch_size*n_channels, n_patches).to(self.device_name)

                with torch.amp.autocast(device_type='cuda'):
                    output = self.model(inputs_embeds = enc_in, attention_mask = attn_mask.to(torch.bool))
                
                enc_out = output.last_hidden_state
                enc_out = enc_out.reshape((-1, n_channels, n_patches, self.d_model))
                mask_out = [enc_out[..., indices, :] for indices in self.SCALE_INDEX] 
                head_out = [head(mask) for head, mask in zip(self.heads,mask_out)]
                denorm_out = [self.normalizer(x=_, mode="denorm") for _ in head_out]

                # upsampling
                up_sampling = [self._upsample(head) for head in denorm_out]

                weighted_forecast = 0
                scale_weights = torch.softmax(self.scale_weights, dim=0)
                for i, weight in enumerate(scale_weights):
                    weighted_forecast += weight * up_sampling[i] 
                
                loss = self.criterion(weighted_forecast.float(), forecast.float().to(self.device_name))
                losses.append(loss.item())
                trues.append(forecast.detach().cpu().numpy())
                preds.append(weighted_forecast.detach().cpu().numpy())
                histories.append(timeseries.detach().cpu().numpy())

            losses = np.array(losses)
            average_loss = np.average(losses)
            trues = np.concatenate(trues, axis=0)
            preds = np.concatenate(preds, axis=0)
            histories = np.concatenate(histories, axis=0)
            
            metrics = get_forecasting_metrics(y=trues, y_hat=preds, reduction="mean")

            writer.add_scalar(f"Loss/Test", average_loss, global_step=cur_epoch)
            writer.add_scalar(f"Test/MSE", metrics.mse, global_step=cur_epoch)
            writer.add_scalar(f"Test/MAE", metrics.mae, global_step=cur_epoch)

    def save_model(self):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }

        with open(os.path.join("./", f"{self.dataset}_{self.pred_length}_checkpoint.pth"),"wb",) as f:
            torch.save(checkpoint, f)
    def main(self):
        self._build_model()
        self._get_dataloader()
        self._get_optimizer()
        self._get_lr_schedular()

        self.train()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ETTh1", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2"])
    parser.add_argument("--train_bs", type=int, default=8, help="Batch size for training")
    parser.add_argument("--eval_bs", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--max_epoch", type=int, default=3, help="Maximum number of training epochs")
    parser.add_argument("--init_lr", type=float, default=1e-4, help="Initial learning rate (default is dataset-specific)")
    parser.add_argument("--head_lr", type=float, default=1e-4, help="Initial learning rate (default is dataset-specific)")
    parser.add_argument("--scale_weight_lr", type=float, default=5e-5, help="Learning rate for scale weights")
    parser.add_argument("--pred_length", type=int, default=96, help="Prediction length")
    parser.add_argument("--lora", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--linear", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--head_dropout", type=float, default=0.1, help="head_dropout")
    parser.add_argument("--patience", type=int, default=5, help="patience")
    parser.add_argument("--version", type=str, default="small", help="")
    parser.add_argument("--note", type=str, default='')
    args = parser.parse_args()
    writer = SummaryWriter(log_dir=f"../tf-logs/runs/{args.note}")
    MomentFinetune(args).main()
    writer.close()