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
        self.num_new_scales = args.num_new_scales
        self.pred_mask_tokens = args.pred_mask_tokens
        self.model_name = args.version
        self.d_model = {"small":512, "base":768, "large":1024}[self.model_name]
        self.seq_len = 512
        self.patch_size = 8
        self.ds_factor = 2 
        self.pred_length = args.pred_length
        self.patch_stride_len = 8
        self.mask_generator = Masking()
        self.early_stopping = EarlyStopping(
            patience=self.patience, 
            verbose=True,
            delta=0.001,
            # path=f"{self.dataset}_{self.pred_length}_best_model.pth",
            trace_func=print
        )
        # 新模块参数
        self.lora = args.lora
        self.linr = args.linear
        self.head_dropout = args.head_dropout

        # Dim of Head's input. Use masked pred tokens or all ctx tokens.
        if self.pred_mask_tokens:
            self.head_nf = [self.d_model * _ for _ in self._num_pred_tokens_per_scale]
        else:
            self.head_nf = [self.d_model * _ for _ in self._num_ctx_tokens_per_scale]
        self.scale_weights = nn.Parameter(torch.ones(1+self.num_new_scales, device=self.device_name))
        self.linear = nn.ModuleList([nn.Linear(self.d_model, self.d_model) for _ in range(1+self.num_new_scales)]).to(self.device_name)
        # 初始化为单位矩阵
        for layer in self.linear:
            nn.init.eye_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # 记录scale_loss的列表
        self.scale_loss_history = [[] for _ in range(1+self.num_new_scales)]
        self.L = []
        
        # 初始化预测头
        pred_length = 2*self.pred_length  # 2*初始预测长度
        self.heads = nn.ModuleList([
            ForecastingHead(_, pred_length, self.head_dropout).to(self.device_name)
            for _ in self.head_nf
            if (pred_length := max(1, pred_length // 2))  # 确保最小值不小于1
        ])

    @property
    def _ctx_len_per_scale(self):
        return [math.ceil(self.seq_len / 2**i) for i in range(1+self.num_new_scales)]

    @property
    def _pred_len_per_scale(self):
        return [math.ceil(self.pred_length / 2 ** i) for i in range(1+self.num_new_scales)]

    @property
    def _num_ctx_tokens_per_scale(self):
        return [math.ceil(self._ctx_len_per_scale[i] / self.patch_size) for i in range(1+self.num_new_scales)]

    @property
    def _num_pred_tokens_per_scale(self):
        return [math.ceil(self._pred_len_per_scale[i] / self.patch_size) for i in range(1+self.num_new_scales)]

    def _index_of_a_given_scale(self, scale):
        start = 0
        if self.pred_mask_tokens:
            for i in range(scale):
                start += self._num_ctx_tokens_per_scale[i] + self._num_pred_tokens_per_scale[i]
            end = start + self._num_ctx_tokens_per_scale[scale] + self._num_pred_tokens_per_scale[scale]
        else:
            for i in range(scale):
                start += self._num_ctx_tokens_per_scale[i]
            end = start + self._num_ctx_tokens_per_scale[scale]
        return list(range(start, end))

    def _pred_index_of_a_given_scale(self, scale):
        start = 0
        for i in range(scale):
            start += self._num_ctx_tokens_per_scale[i] + self._num_pred_tokens_per_scale[i]
        start = start + self._num_ctx_tokens_per_scale[scale]
        end = start + self._num_pred_tokens_per_scale[scale]
        return list(range(start, end))

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
        self.scale_normalizers = nn.ModuleList([RevIN(num_features=1, affine=False) for _ in range(1+self.num_new_scales)])
        print(config)
        if self.lora:
            self.model =  T5EncoderModel_LoRA(config,
                                              num_new_scales=self.num_new_scales,
                                              pred_length=self.pred_length,
                                              pred_mask_tokens=self.pred_mask_tokens
                                              ).get_encoder()
        else:
            self.model =  T5EncoderModel(config).get_encoder()
        checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"MOMENT_{self.model_name}.ckpt")
        self.checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        encoder_state_dict = {k.replace("encoder.", ""): v for k, v in self.checkpoint.items() if k.startswith("encoder.")}
        missing_keys, unexpected_keys = self.model.load_state_dict(encoder_state_dict, strict=False)

        self._get_embeder()
        self.model.to(self.device_name)

        # 冻结训练参数
        for name, param in self.model.named_parameters():
            # 冻结所有 self-attention 层的 base_layer  # SelfAttention.o
            if "SelfAttention.q" in name or "SelfAttention.k" in name or "SelfAttention.v" in name or "DenseReluDense" in name:
                param.requires_grad = False

    def _get_embeder(self):
        self.patch_embedding = PatchEmbedding(
            d_model=self.d_model,
            seq_len=self.seq_len,
            patch_len=self.patch_size,
            stride=self.patch_stride_len,
            patch_dropout=0.1,
            add_positional_embedding=True,
            value_embedding_bias=False,
            orth_gain=1.41,
        )

        # 重新映射 checkpoint 里的 key,加载到PatchEmbedding中
        patch_embedding_checkpoint = {key.replace("patch_embedding.", ""): value for key, value in self.checkpoint.items()}
        self.patch_embedding.load_state_dict(patch_embedding_checkpoint, strict=False)
        self.patch_embedding = self.freeze_parameters(self.patch_embedding)
        self.patch_embedding.to(self.device_name)

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

        if 'ETT' in self.dataset:
            prefix = 'ETT-small'
        else:
            prefix = self.dataset

        train_dataset = InformerDataset(data_split="train", random_seed=13, forecast_horizon=self.pred_length,
                                        full_file_path_and_name=f"./long_term_forecast/{prefix}/{self.dataset}.csv")
        val_dataset = InformerDataset(data_split="val", random_seed=13, forecast_horizon=self.pred_length,
                                      full_file_path_and_name=f"./long_term_forecast/{prefix}/{self.dataset}.csv")
        test_dataset = InformerDataset(data_split="test", random_seed=13, forecast_horizon=self.pred_length,
                                       full_file_path_and_name=f"./long_term_forecast/{prefix}/{self.dataset}.csv")

        train_loader = DataLoader(train_dataset, batch_size=self.train_bs, num_workers=self.NumWorkers, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.eval_bs, num_workers=self.NumWorkers, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.eval_bs, num_workers=self.NumWorkers, shuffle=True, pin_memory=True)
        self.dataloader = {"train":train_loader, "val":val_loader, "test":test_loader}

    def _downsample(self, timeseries, forecast, input_mask) -> tuple:
        """
        timeseries: [batch_size, channel, context_length]
        forecast: [batch_size, channel, predict_length]
        input_mask: [batch_size, context_length]

        Returns:
        scale_ts: List of (bs, channel, num_patch, ps) scale_fc: List of (bs, channel, predict_length)
        scale_input_mask: (batch_size, combined_sequence_length)
        scale_pred_mask: 在一个patch sequence中表示哪些是需要预测的[batch_size, num_patch]
        """
        batch_size, channel, context_length = timeseries.shape
        _, _, prediction_length = forecast.shape
        scale_ts, scale_fc, scale_input_mask, scale_pred_mask = [], [], [], []

        # timeseries = self.normalizer(x=timeseries, mask=input_mask, mode="norm")

        timeseries = torch.nan_to_num(timeseries, nan=0, posinf=0, neginf=0)
        new_context_length = context_length
        new_prediction_length = prediction_length
        new_timeseries = timeseries
        new_forecast = forecast
        new_input_mask = input_mask

        for i in range(1+self.num_new_scales):
            self.ds_factor = 1 if i == 0 else 2
            new_context_length = math.ceil(new_context_length / self.ds_factor)
            new_prediction_length = math.ceil(new_prediction_length / self.ds_factor)

            # QZ: 每个scale更新input_mask
            new_input_mask = new_input_mask.unsqueeze(1).float()
            new_input_mask = nn.functional.max_pool1d(new_input_mask, kernel_size=self.ds_factor, stride=self.ds_factor)
            new_input_mask = new_input_mask.squeeze(1).int()
            # 每个scale单独做RevIN
            new_timeseries = nn.functional.avg_pool1d(new_timeseries, kernel_size=self.ds_factor, stride=self.ds_factor)
            new_normed_timeseries = self.scale_normalizers[i](x=new_timeseries, mask=new_input_mask, mode="norm")

            new_forecast = nn.functional.avg_pool1d(new_forecast, kernel_size=self.ds_factor, stride=self.ds_factor)
            scale_fc.append(new_forecast)

            if new_prediction_length % self.patch_size != 0:
                padding_needed = self.patch_size - (new_prediction_length % self.patch_size)
                new_forecast = torch.nn.functional.pad(new_forecast, (0, padding_needed))

            if self.pred_mask_tokens:  # 拼接context和预测部分
                ts_patch = new_normed_timeseries.reshape(batch_size, channel, -1, self.patch_size)
                fc_patch = new_forecast.reshape(batch_size, channel, -1, self.patch_size)
                combined = torch.cat([ts_patch, fc_patch], dim=2).to(torch.float32)
                scale_ts.append(combined)

                # 生成scale_pred_mask
                ones = torch.ones(batch_size, ts_patch.shape[2], device=new_input_mask.device)
                zeros = torch.zeros(batch_size, fc_patch.shape[2], device=new_input_mask.device)
                pred_mask = torch.cat([ones, zeros], dim=1)
                scale_pred_mask.append(pred_mask)

                ones = torch.ones((batch_size, new_forecast.size(-1)), device=new_input_mask.device)
                combined = torch.cat((new_input_mask, ones), dim=1)
                scale_input_mask.append(combined)

            else:
                ts_patch = new_normed_timeseries.reshape(batch_size, channel, -1, self.patch_size).float()
                scale_ts.append(ts_patch)

                scale_pred_mask.append(torch.ones((ts_patch.size(0), ts_patch.size(2)), device=ts_patch.device))

                scale_input_mask.append(new_input_mask)

        return scale_ts, scale_fc, torch.concat(scale_input_mask, dim=1), scale_pred_mask
        
    def _upsample(self,  input_tensor) -> tuple:
        batch_size, n_channel, length = input_tensor.shape
        
        factor = self.pred_length // length
        upsampled_tensor = input_tensor.repeat_interleave(factor, dim=2)
        return upsampled_tensor

    def freeze_parameters(self, model):
        """
        Freeze parameters of the model
        ❄️:patch_embedding/encoder
        """
        # Freeze the parameters
        for name, param in model.named_parameters():
            param.requires_grad = False

        return model
    
    def embed(self, x, masks):
        """
        masks: If replace tokens with mask embeddings
        """
        input_embed = [self.patch_embedding(_, mask) for _, mask in zip(x, masks)]
        return input_embed


    # def embed(self, x, masks):
    #     """
    #     masks: If replace tokens with mask embeddings
    #     """
    #     x_ = torch.cat(x, dim=2)
    #     masks_ = torch.cat(masks, dim=1)
    #     input_embed = self.patch_embedding(x_, masks_)
    #
    #     scale_input_embed = []
    #
    #     for i in range(1+self.num_new_scales):
    #         scale_input_embed.append(input_embed[:, :, self._index_of_a_given_scale(i), :])
    #
    #     return scale_input_embed



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

    def forward_batch(self, timeseries, forecast, input_mask):
        timeseries = timeseries.to(self.device_name)
        forecast = forecast.to(self.device_name)
        input_mask = input_mask.to(self.device_name)

        batch_size, n_channels, _ = timeseries.shape

        scale_ts, scale_fc, observed_mask, scale_pred_mask = self._downsample(timeseries, forecast, input_mask)
        n_patches = sum([_.shape[2] for _ in scale_ts])
        input_embed = self.embed(scale_ts, scale_pred_mask)  # List of (bs, channel, patch, ps)

        norm0 = self.scale_normalizers[0]
        norm1 = self.scale_normalizers[1]

        if self.linr:
            for i in range(1 + self.num_new_scales):
                if self.pred_mask_tokens:  # 只对context部分投影
                    ctx_tokens = self._num_ctx_tokens_per_scale[i]
                    input_embed[i][:, :, :ctx_tokens, :] = self.linear[i](input_embed[i][:, :, :ctx_tokens, :])
                else:
                    input_embed[i] = self.linear[i](input_embed[i])

        # 拼接不同尺度
        enc_in = torch.cat(input_embed, dim=2).reshape(batch_size * n_channels, n_patches, self.d_model)

        # attention mask
        patch_view_mask = Masking.convert_seq_to_patch_view(observed_mask, self.patch_size).to(self.device_name)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            output = self.model(inputs_embeds=enc_in, attention_mask=attention_mask)
            enc_out = output.last_hidden_state.reshape(-1, n_channels, n_patches, self.d_model)
            if self.pred_mask_tokens:
                scale_repr = [enc_out[..., self._pred_index_of_a_given_scale(i), :] for i in range(1 + self.num_new_scales)]
            else:
                scale_repr = [enc_out[..., self._index_of_a_given_scale(i), :] for i in range(1 + self.num_new_scales)]
            scale_head_out = [head(repr) for head, repr in zip(self.heads, scale_repr)]
            scale_denorm_out = [self.scale_normalizers[i](x=scale_head_out[i], mode="denorm") for i in range(1 + self.num_new_scales)]

        return scale_denorm_out, scale_fc

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
                denorm_out, scale_fc = self.forward_batch(timeseries, forecast, input_mask)

                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
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

                if step % 100 == 0:
                    avg_loss = np.mean(losses[-100:])  # 计算最近 100 个步骤的平均损失
                    writer.add_scalar("Loss/train_step", avg_loss, step)
                    print(f"Epoch [{epoch + 1}/{self.max_epoch}] Step [{step}] - Train Loss: {avg_loss:.3f}")

            train_loss = np.mean(losses)
            writer.add_scalar("Loss/train", train_loss, global_step=epoch)

            scale_weights = torch.softmax(self.scale_weights, dim=0)
            for i, weight in enumerate(scale_weights):
                writer.add_scalar(f'scale_weights/{i}', weight, global_step=epoch)

            self.lr_scheduler.step()

            val_loss = self.eval(cur_epoch=epoch, mode='val')

            if self.BestEvalLoss > val_loss:
                self.BestEvalLoss = val_loss
                self.eval(cur_epoch=epoch, mode='test')
                # self.save_model()

            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                print(f"Training stopped early at epoch {epoch + 1}/{self.max_epoch}")
                break
    
    @torch.no_grad()
    def eval(self, cur_epoch, mode="val"):
        trues, preds, histories, losses = [], [], [], []
        self.model.eval()
        self.patch_embedding.eval()
        self.linear.eval()
        self.heads.eval()

        for timeseries, forecast, input_mask in tqdm(self.dataloader[f"{mode}"], total=len(self.dataloader[f"{mode}"])):

            denorm_out, _ = self.forward_batch(timeseries, forecast, input_mask)
            up_sampling = [self._upsample(out) for out in denorm_out]

            weighted_forecast = 0
            scale_weights = torch.softmax(self.scale_weights, dim=0)
            for i, weight in enumerate(scale_weights):
                weighted_forecast += weight * up_sampling[i]

            loss = self.criterion(weighted_forecast.float(), forecast.float().to(self.device_name))
            losses.append(loss.item())
            trues.append(forecast.detach().cpu().numpy())
            preds.append(weighted_forecast.detach().cpu().numpy())

        losses = np.array(losses)
        average_loss = np.average(losses)
        trues = np.concatenate(trues, axis=0)
        preds = np.concatenate(preds, axis=0)

        metrics = get_forecasting_metrics(y=trues, y_hat=preds, reduction="mean")

        writer.add_scalar(f"Loss/{mode}", average_loss, global_step=cur_epoch)
        writer.add_scalar(f"{mode}/MSE", metrics.mse, global_step=cur_epoch)
        writer.add_scalar(f"{mode}/MAE", metrics.mae, global_step=cur_epoch)

        return average_loss

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
    parser.add_argument("--dataset", type=str, default="ETTh1")
    parser.add_argument("--num_new_scales", type=int, default=1, help="Batch size for training")
    parser.add_argument("--train_bs", type=int, default=64, help="Batch size for training")
    parser.add_argument("--eval_bs", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--max_epoch", type=int, default=10, help="Maximum number of training epochs")
    parser.add_argument("--init_lr", type=float, default=5e-7, help="Initial learning rate (default is dataset-specific)")
    parser.add_argument("--head_lr", type=float, default=1e-3, help="Initial learning rate (default is dataset-specific)")
    parser.add_argument("--scale_weight_lr", type=float, default=1e-2, help="Learning rate for scale weights")
    parser.add_argument("--pred_length", type=int, default=96, help="Prediction length")
    parser.add_argument("--lora", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--linear", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--pred_mask_tokens", type=lambda x: x.lower() == "true", default=False,
                        help="Use masked prediction tokens like Moirai or Use context tokens like original Moment")
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--head_dropout", type=float, default=0.1, help="head_dropout")
    parser.add_argument("--patience", type=int, default=5, help="patience")
    parser.add_argument("--version", type=str, default="small", help="")
    parser.add_argument("--note", type=str, default='')
    args = parser.parse_args()
    writer = SummaryWriter(log_dir=f"./tf-logs/{args.version}/{args.dataset}/{args.note}")
    MomentFinetune(args).main()
    writer.close()