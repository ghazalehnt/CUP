import json
import operator
import os
import random
import time
from os.path import exists, join

import torch
from datasets import Dataset
from ray import tune
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from SBR.utils.metrics import calculate_metrics_per_user, log_results
from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD
from SBR.utils.data_loading import CollateUserItem


class SupervisedTrainer:
    def __init__(self, config, model, device, logger, exp_dir, test_only=False, tuning=False, save_checkpoint=True,
                     relevance_level=1, users=None, items=None, dataset_eval_neg_sampling=None, to_load_model_name=None,
                 padding_token=None, unique_user_item=None):
        self.model = model
        self.device = device
        self.logger = logger
        self.padding_token = padding_token
        self.unique_u_i = unique_user_item
        self.test_only = test_only  # todo used?
        self.tuning = tuning
        self.save_checkpoint = save_checkpoint
        self.relevance_level = relevance_level
        self.valid_metric = config['valid_metric']
        self.patience = config['early_stopping_patience'] if ('early_stopping_patience' in config and config['early_stopping_patience'] != '') else None
        self.do_validation = config["do_validation"]
        self.best_model_train_path = None
        self.last_model_path = None
        self.best_model_path = join(exp_dir, 'best_model.pth')
        if "save_best_train" in config and config["save_best_train"] is True:
            self.best_model_train_path = join(exp_dir, 'best_model_tr_loss.pth')
        if "save_every_epoch" in config and config["save_every_epoch"] is True:
            self.last_model_path = join(exp_dir, 'last_model.pth')
        # if a model was not given, load best model (valid)
        if to_load_model_name is not None:
            self.to_load_model = join(exp_dir, f"{to_load_model_name}.pth")
        else:
            self.to_load_model = self.best_model_path
            to_load_model_name = "best_model"

        neg_name = dataset_eval_neg_sampling['test']
        if neg_name.startswith("f:"):
            neg_name = neg_name[len("f:"):]
        self.test_output_path = {"ground_truth": join(exp_dir, f'test_ground_truth_{neg_name}.json'),
                                 "predicted": join(exp_dir, f'test_predicted_{neg_name}_{to_load_model_name}')}

        self.users = users
        self.items = items
        self.sig_output = config["sigmoid_output"]
        self.enable_autocast = False
        self.validation_user_sample_num = None
        if "enable_autocast" in config:
            self.enable_autocast = config["enable_autocast"]
        if "validation_user_sample_num" in config and config["validation_user_sample_num"] != "":
            self.validation_user_sample_num = config["validation_user_sample_num"]

        if config['loss_fn'] == "BCE":
            if self.sig_output is False:
                raise ValueError("cannot have BCE with no sigmoid")
            self.loss_fn = torch.nn.BCEWithLogitsLoss()  # use BCEWithLogitsLoss and do not apply the sigmoid beforehand
        elif config['loss_fn'] == "MRL":
            self.loss_fn = torch.nn.MarginRankingLoss(margin=config["margin"])
        elif config['loss_fn'] == "MSE":
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError(f"loss_fn {config['loss_fn']} is not implemented!")

        self.epochs = config['epochs']
        self.start_epoch = 0
        self.best_epoch = 0
        self.best_saved_valid_metric = np.inf if self.valid_metric == "valid_loss" else -np.inf
        if exists(self.to_load_model):
            checkpoint = torch.load(self.to_load_model, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_epoch = checkpoint['epoch']
            if "best_valid_metric" in checkpoint:
                self.best_saved_valid_metric = checkpoint['best_valid_metric']
            print("last checkpoint restored")
        self.model.to(device)

        if not test_only:
            if "bert_lr" in config:
                bert_params = self.model.bert.parameters()
                bert_param_names = [f"bert.{k[0]}" for k in self.model.bert.named_parameters()]
                other = [v for k, v in self.model.named_parameters() if k not in bert_param_names]
                opt_params = [
                    {'params': other},
                    {'params': bert_params, 'lr': config["bert_lr"]}]
            else:
                opt_params = self.model.parameters()
            if config['optimizer'] == "Adam":
                self.optimizer = Adam(opt_params, lr=config['lr'], weight_decay=config['wd'])
            elif config['optimizer'] == "AdamW":
                self.optimizer = AdamW(opt_params, lr=config['lr'], weight_decay=config['wd'])
            elif config['optimizer'] == "SGD":
                self.optimizer = SGD(opt_params, lr=config['lr'], weight_decay=config['wd'], momentum=config['momentum'], nesterov=config['nesterov'])
            else:
                raise ValueError(f"Optimizer {config['optimizer']} not implemented!")
            if exists(self.to_load_model):
                if "optimizer_state_dict" in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                else:
                    print("optimizer_state_dict was not saved in the checkpoint")

    def fit(self, train_dataloader, valid_dataloader):
        early_stopping_cnt = 0
        comparison_op = operator.lt if self.valid_metric == "valid_loss" else operator.gt

        if self.validation_user_sample_num is not None:
            valid_dataset_pd = valid_dataloader.dataset.to_pandas()
            valid_users = list(set(valid_dataset_pd[INTERNAL_USER_ID_FIELD]))

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        # Creates a GradScaler once at the beginning of training.
        scaler = GradScaler()

        best_train_loss = np.inf
        for epoch in range(self.start_epoch, self.epochs):
            if self.patience is not None and early_stopping_cnt == self.patience:
                print(f"Early stopping after {self.patience} epochs not improving!")
                break

            self.model.train()
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=True if self.tuning else False)
            start_time = time.perf_counter()
            train_loss, total_count = 0, 0

            # for loop going through dataset
            for batch_idx, batch in pbar:
                # data preparation
                user_index, item_index = None, None
                if self.unique_u_i:
                    temp = batch.pop('item_index')
                    item_index = {temp[i]: i for i in range(len(temp))}
                    temp = batch.pop('user_index')
                    user_index = {temp[i]: i for i in range(len(temp))}

                batch = {k: v.to(self.device) for k, v in batch.items()}
                label = batch.pop("label").float()  # setting the type to torch.float32
                # print(f"{min(label)} {max(label)}")
                prepare_time = time.perf_counter() - start_time

                self.optimizer.zero_grad()
                # Runs the forward pass with autocasting.
                with autocast(enabled=self.enable_autocast, device_type='cuda', dtype=torch.float16):
                    if self.unique_u_i:
                        output = self.model(batch, user_index=user_index, item_index=item_index)
                    else:
                        output = self.model(batch)
                    if self.loss_fn._get_name() == "BCEWithLogitsLoss":
                        # not applying sigmoid before loss bc it is already applied in the loss
                        loss = self.loss_fn(output, label)
                    else:
                        if self.sig_output:
                            output = torch.sigmoid(output)
                        if self.loss_fn._get_name() == "MarginRankingLoss":
                            pos_idx = set((label == 1).nonzero(as_tuple=True)[0].tolist())
                            x1 = []
                            x2 = []
                            for uid in set([k[0] for k in batch[INTERNAL_USER_ID_FIELD].tolist()]):
                                u_idxs = set((batch[INTERNAL_USER_ID_FIELD] == uid).nonzero(as_tuple=True)[0].tolist())
                                pos_u_idx = u_idxs.intersection(pos_idx)
                                neg_u_idx = u_idxs - pos_u_idx
                                for pos in pos_u_idx:
                                    for neg in neg_u_idx:
                                        x1.append(pos)
                                        x2.append(neg)
                            loss = self.loss_fn(output[x1], output[x2], torch.ones((len(x1), 1), device=self.device))
                        else:
                            loss = self.loss_fn(output, label)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                train_loss += loss
                total_count += label.size(0)

                # compute computation time and *compute_efficiency*
                process_time = time.perf_counter() - start_time - prepare_time
                compute_efficiency = process_time / (process_time + prepare_time)
                pbar.set_description(
                    f'Compute efficiency: {compute_efficiency:.4f}, '
                    f'loss: {loss.item():.8f},  epoch: {epoch}/{self.epochs}'
                    f'prep: {prepare_time:.4f}, process: {process_time:.4f}')
                start_time = time.perf_counter()
            train_loss /= total_count
            print(f"Train loss epoch {epoch}: {train_loss}")
            if self.best_model_train_path is not None:
                if train_loss < best_train_loss:
                    checkpoint = {
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }
                    torch.save(checkpoint, f"{self.best_model_train_path}_tmp")
                    os.rename(f"{self.best_model_train_path}_tmp", self.best_model_train_path)
                    best_train_loss = train_loss
            if self.last_model_path is not None:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
                torch.save(checkpoint, f"{self.last_model_path}_tmp")
                os.rename(f"{self.last_model_path}_tmp", self.last_model_path)

            # udpate tensorboardX  TODO for logging use what  mlflow, files, tensorboard
            self.logger.add_scalar('epoch_metrics/epoch', epoch, epoch)
            self.logger.add_scalar('epoch_metrics/train_loss', train_loss, epoch)

            if self.do_validation:
                if self.validation_user_sample_num is not None:
                    chosen_users = np.random.choice(valid_users, self.validation_user_sample_num, replace=False)
                    sampled_validation = Dataset.from_pandas(
                        valid_dataset_pd[valid_dataset_pd[INTERNAL_USER_ID_FIELD].isin(chosen_users)], preserve_index=False)
                    sampled_dataloader = DataLoader(sampled_validation,
                                                    batch_size=valid_dataloader.batch_size,
                                                    collate_fn=valid_dataloader.collate_fn,
                                                    num_workers=valid_dataloader.num_workers)
                    outputs, ground_truth, valid_loss, users, items = self.predict(sampled_dataloader, low_mem=True)
                else:
                    start_time = time.perf_counter()
                    if hasattr(self.model, 'support_test_prec') and self.model.support_test_prec is True:
                        self.model.prec_representations_for_test(self.users, self.items,
                                                                 padding_token=self.padding_token)
                    outputs, ground_truth, valid_loss, users, items = self.predict(valid_dataloader, low_mem=True)
                    self.model.user_prec_reps = None
                    self.model.item_prec_reps = None

                results = calculate_metrics_per_user(ground_truth, outputs, users, items, self.relevance_level)
                results["loss"] = valid_loss
                results = {f"valid_{k}": v for k, v in results.items()}
                for k, v in results.items():
                    self.logger.add_scalar(f'epoch_metrics/{k}', v, epoch)
                print(f"Valid loss epoch {epoch}: {valid_loss} - {self.valid_metric} = {results[self.valid_metric]:.6f} done in {(time.perf_counter() - start_time)/60}\n")

                if comparison_op(results[self.valid_metric], self.best_saved_valid_metric):
                    self.best_saved_valid_metric = results[self.valid_metric]
                    self.best_epoch = epoch
                    if self.save_checkpoint:
                        checkpoint = {
                            'epoch': self.best_epoch,
                            'best_valid_metric': self.best_saved_valid_metric,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            }
                        torch.save(checkpoint, f"{self.best_model_path}_tmp")
                        os.rename(f"{self.best_model_path}_tmp", self.best_model_path)
                    early_stopping_cnt = 0
                else:
                    early_stopping_cnt += 1
                self.logger.add_scalar('epoch_metrics/best_epoch', self.best_epoch, epoch)
                self.logger.add_scalar('epoch_metrics/best_valid_metric', self.best_saved_valid_metric, epoch)

                # report to tune
                if self.tuning:
                    tune.report(best_valid_metric=self.best_saved_valid_metric,
                                best_epoch=self.best_epoch,
                                epoch=epoch)

            self.logger.flush()

    def evaluate_dataloader(self, eval_dataloader, eval_output_path):
        # load the best model from file.
        # because we may call evaluate right after fit and in this case need to reload the best model!
        checkpoint = torch.load(self.to_load_model, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.best_epoch = checkpoint['epoch']
        print("best model loaded!")
        outfile = f"{eval_output_path['predicted']}_e-{self.best_epoch}.json"

        if hasattr(self.model, 'support_test_prec') and self.model.support_test_prec is True:
            self.model.prec_representations_for_test(self.users, self.items, padding_token=self.padding_token)
            outfile = f"{eval_output_path['predicted']}_p_e-{self.best_epoch}.json"

        outputs, ground_truth, loss, internal_user_ids, internal_item_ids = self.predict(eval_dataloader)
        log_results(ground_truth, outputs, internal_user_ids, internal_item_ids,
                    self.users, self.items,
                    eval_output_path['ground_truth'],
                    outfile,
                    f"{eval_output_path['log']}_e-{self.best_epoch}.txt" if "log" in eval_output_path else None)

    def evaluate(self, test_dataloader, valid_dataloader):
        self.evaluate_dataloader(test_dataloader, self.test_output_path)


    def predict(self, eval_dataloader, low_mem=False):
        # bring models to evaluation mode
        self.model.eval()

        outputs = []
        ground_truth = []
        user_ids = []
        item_ids = []
        eval_loss, total_count = 0, 0
        pbar = tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), disable=True if self.tuning else False)

        start_time = time.perf_counter()
        # log_user_texts = {}
        with torch.no_grad():
            for batch_idx, batch in pbar:
                # data preparation
                user_index, item_index = None, None
                if self.unique_u_i:
                    temp = batch.pop('item_index')
                    item_index = {temp[i]: i for i in range(len(temp))}
                    temp = batch.pop('user_index')
                    user_index = {temp[i]: i for i in range(len(temp))}

                batch = {k: v.to(self.device) for k, v in batch.items()}
                label = batch.pop("label").float()  # setting the type to torch.float32
                prepare_time = time.perf_counter() - start_time

                with autocast(enabled=self.enable_autocast, device_type='cuda', dtype=torch.float16):
                    if hasattr(self.model, 'support_test_prec') and self.model.support_test_prec is True and self.model.user_prec_reps is not None:
                        if self.unique_u_i:
                            output = self.model(batch, user_index=user_index, item_index=item_index, validate=True)
                        else:
                            output = self.model(batch, validate=True)
                    else:
                        if self.unique_u_i:
                            output = self.model(batch, user_index=user_index, item_index=item_index)
                        else:
                            output = self.model(batch)

                    if self.loss_fn._get_name() == "BCEWithLogitsLoss":
                        # not applying sigmoid before loss bc it is already applied in the loss
                        loss = self.loss_fn(output, label)
                        # just apply sigmoid for logging
                        output = torch.sigmoid(output)
                    else:
                        if self.sig_output:
                            output = torch.sigmoid(output)
                        if self.loss_fn._get_name() == "MarginRankingLoss":
                            loss = torch.Tensor([-1])  # cannot calculate margin loss with more than 1 negative per positve
                        else:
                            loss = self.loss_fn(output, label)

                eval_loss += loss.item()
                total_count += label.size(0)  # TODO remove if not used
                process_time = time.perf_counter() - start_time - prepare_time
                proc_compute_efficiency = process_time / (process_time + prepare_time)

                ## for debugging. it needs access to actual user_id and item_id
                ground_truth.extend(label.squeeze(1).tolist())
                outputs.extend(output.squeeze(1).tolist())
                user_ids.extend(batch[
                                    INTERNAL_USER_ID_FIELD].squeeze(1).tolist())
                if not low_mem:
                    item_ids.extend(batch[INTERNAL_ITEM_ID_FIELD].squeeze(1).tolist())

                postprocess_time = time.perf_counter() - start_time - prepare_time - process_time
                pbar.set_description(
                    f'Compute efficiency: {proc_compute_efficiency:.4f}, '
                    f'loss: {loss.item():.8f},  prep: {prepare_time:.4f},'
                    f'process: {process_time:.4f}, post: {postprocess_time:.4f}')
                start_time = time.perf_counter()

            eval_loss /= total_count
        # print(log_user_texts)
        ground_truth = torch.tensor(ground_truth)
        outputs = torch.tensor(outputs)
        return outputs, ground_truth, eval_loss, user_ids, item_ids

    def log(self):
        self.model.prec_representations_for_test(self.users, self.items, padding_token=self.padding_token)
        torch.save(self.model.user_prec_reps, self.user_prec_file_out)
        torch.save(self.model.item_prec_reps, self.item_prec_file_out)
