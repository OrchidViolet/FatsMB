
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .models import DotProductPredictionHead
from .models.ldm4rec import LDM
from .utils import recalls_and_ndcgs_for_ks

from datetime import datetime

import pdb

class RecModel(pl.LightningModule):
    def __init__(self,
            backbone: LDM,
            b_head: bool = False,
            pretrain_epoch: int = None,
            diffusion_epoch: int = None,
            mutual_training: bool = False,
            epoch_output: bool = True,
        ):
        super().__init__()
        self.backbone = backbone
        self.num_items = backbone.num_items
        self.n_b = backbone.n_b
        self.max_len = backbone.max_len
        self.head = DotProductPredictionHead(backbone.d_model, backbone.num_items, self.backbone.embedding.token)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.pretrain_epoch = pretrain_epoch
        self.diffusion_epoch = diffusion_epoch
        self.mutual_training = mutual_training

        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(self.start_time)
        
        self.log_path = f'./logs/{self.backbone.__class__.__name__}_pretrain_epoch_{self.pretrain_epoch}_diff_steps_{self.backbone.diff_steps}_{self.start_time}.log'
        self.best_recall_20 = float('-inf')
        self.best_ndcg_20 = float('-inf')
        self.best_recall_10 = float('-inf')
        self.best_ndcg_10 = float('-inf')

        self.epoch_output = epoch_output

    def on_train_epoch_start(self):
        if self.current_epoch == self.pretrain_epoch:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.head.parameters():
                param.requires_grad = False
            for param in self.backbone.diff.parameters():
                param.requires_grad = True
        if self.current_epoch == self.diffusion_epoch:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.head.out.parameters():
                param.requires_grad = True
            self.head.bias.requires_grad = True

    def training_step(self, batch, batch_idx):
        if self.current_epoch < self.pretrain_epoch:
            input_ids = batch['input_ids']
            b_seq = batch['behaviors']
            labels = batch['labels']

            outputs = self.backbone.bert_encode_mask(input_ids, b_seq, labels)
            outputs = outputs.view(-1, outputs.size(-1))

            labels = labels.view(-1)
            valid = labels > 0
            valid_index = valid.nonzero().squeeze()
            valid_outputs = outputs[valid_index]
            valid_b_seq = b_seq.view(-1)[valid_index]
            valid_labels = labels[valid_index]

            valid_logits = self.head(valid_outputs, valid_b_seq)
            loss = self.loss(valid_logits, valid_labels)
            self.log('train_loss', loss.detach().item(), on_step=True, on_epoch=True, prog_bar=True)

            return {'loss': loss}

        elif self.pretrain_epoch <= self.current_epoch < self.diffusion_epoch:
            input_ids = batch['input_ids']
            b_seq = batch['behaviors']
            labels = batch['labels']

            outputs, loss = self.backbone.diffusion_forward_behavior(input_ids, b_seq, labels=labels)
            self.log('train_loss', loss.detach().item(), on_step=True, on_epoch=True, prog_bar=True)

            return {'loss': loss}

        else:
            input_ids = batch['input_ids']
            b_seq = batch['behaviors']
            labels = batch['labels']

            outputs = self.backbone.diffusion_encode_behavior(input_ids, b_seq, labels=labels)

            labels = labels.view(-1)  # BT
            valid = labels > 0
            valid_index = valid.nonzero().squeeze()  # M
            valid_b_seq = b_seq.view(-1)[valid_index]  # M
            valid_logits = self.head(outputs, valid_b_seq)  # M
            valid_labels = labels[valid_index]

            loss = self.loss(valid_logits, valid_labels)
            self.log('train_loss', loss.detach().item(), on_step=True, on_epoch=True, prog_bar=True)

            return {'loss': loss}


    def validation_step(self, batch, batch_idx):
        if self.current_epoch < self.pretrain_epoch:
            input_ids = batch['input_ids']
            b_seq = batch['behaviors']

            outputs = self.backbone.bert_encode(input_ids, b_seq)
            last_outputs = outputs[:, -1, :]  # B x H

            last_b_seq = b_seq[:, -1]
            answer = batch['answer'].squeeze()
            logits = self.head(last_outputs, last_b_seq)[:,1:]
            labels = F.one_hot(answer - 1, num_classes=self.num_items)
            metrics = recalls_and_ndcgs_for_ks(logits, labels, [1, 5, 10, 20, 50])
            
            for k in metrics.keys():
                self.log(f'Val:{k}', metrics[k], on_epoch=True, sync_dist=True)
            
            return metrics

        else:
            input_ids = batch['input_ids']
            b_seq = batch['behaviors']
            index = torch.zeros_like(b_seq)
            index[torch.arange(index.size(0)), -1] = 1

            outputs = self.backbone.diffusion_encode_behavior(input_ids, b_seq, index=index.bool())

            last_b_seq = b_seq[:, -1]
            answer = batch['answer'].squeeze()
            logits = self.head(outputs, last_b_seq)[:,1:]
            labels = F.one_hot(answer - 1, num_classes=self.num_items)
            metrics = recalls_and_ndcgs_for_ks(logits, labels, [1, 5, 10, 20, 50])

            for k in metrics.keys():
                self.log(f'Val:{k}', metrics[k], on_epoch=True, sync_dist=True)
            
            return metrics

    def on_validation_epoch_end(self):
        if self.trainer.global_rank == 0:
            validation_metrics = self.trainer.callback_metrics
            val_metrics = {k: v.detach().item() for k, v in validation_metrics.items() if 
                    k.startswith('Val') and not k.endswith('epoch')}

            print('\n', val_metrics)

            with open(self.log_path, 'a') as f:
                f.write(f'Epoch {self.current_epoch}: \n{val_metrics}\n----------\n')
            self.best_recall_20 = max(self.best_recall_20, val_metrics['Val:Recall@20'])
            self.best_ndcg_20 = max(self.best_ndcg_20, val_metrics['Val:NDCG@20'])
            self.best_recall_10 = max(self.best_recall_10, val_metrics['Val:Recall@10'])
            self.best_ndcg_10 = max(self.best_ndcg_10, val_metrics['Val:NDCG@10'])

    def on_train_end(self):
        print(f'Best Recall@20: {self.best_recall_20}')
        print(f'Best NDCG@20: {self.best_ndcg_20}')
        print(f'Best Recall@10: {self.best_recall_10}')
        print(f'Best NDCG@10: {self.best_ndcg_10}')

        with open(self.log_path, 'a') as f:
            f.write(f'Best Recall@20: {self.best_recall_20}\n')
            f.write(f'Best NDCG@20: {self.best_ndcg_20}\n')
            f.write(f'Best Recall@10: {self.best_recall_10}\n')
            f.write(f'Best NDCG@10: {self.best_ndcg_10}\n')
