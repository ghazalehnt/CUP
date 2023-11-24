import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm

from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD
from SBR.utils.data_loading import CollateUserItem


# DEBUG = True

class BertFFNUserTextProfileItemTextProfileEndToEnd(torch.nn.Module):
    def __init__(self, model_config, device, dataset_config, users, items, test_only):
        super(BertFFNUserTextProfileItemTextProfileEndToEnd, self).__init__()
        self.support_test_prec = True
        self.item_prec_reps = None
        self.user_prec_reps = None
        self.test_only = test_only

        self.agg_strategy = model_config['agg_strategy']
        self.device = device
        self.append_cf_after_ffn = model_config["append_CF_after_ffn"] if "append_CF_after_ffn" in model_config else False

        if dataset_config['max_num_chunks_user'] > 1 or dataset_config['max_num_chunks_item'] > 1:
            raise ValueError("max chunk should be set to 1 ")

        self.bert = transformers.AutoModel.from_pretrained(model_config['pretrained_model'])
        if model_config["tune_BERT"] is True:
            self.bert.requires_grad_(True)
            # freeze layers other than last one:
            freeze_modules = [self.bert.embeddings, self.bert.encoder.layer[:-1]]
            for module in freeze_modules:
                for param in module.parameters():
                    param.requires_grad = False
        else:
            self.bert.requires_grad_(False)
        self.bert_embeddings = self.bert.get_input_embeddings()
        BERT_DIM = self.bert_embeddings.embedding_dim

        dim1user = dim1item = BERT_DIM
        if self.append_cf_after_ffn:
            CF_model_weights = torch.load(model_config['append_CF_after_model_path'], map_location="cpu")[
                'model_state_dict']
            # note: no need to match item and user ids only due to them being created with the same framework where we sort ids.
            # otherwise there needs to be a matching
            self.user_embedding_CF = torch.nn.Embedding.from_pretrained(CF_model_weights['user_embedding.weight'],
                                                                        freeze=model_config["freeze_CF"] if "freeze_CF" in model_config else True)
            self.item_embedding_CF = torch.nn.Embedding.from_pretrained(CF_model_weights['item_embedding.weight'],
                                                                        freeze=model_config["freeze_CF"] if "freeze_CF" in model_config else True)

        user_layers = [torch.nn.Linear(dim1user, model_config["user_k"][0], device=self.device)]
        for k in range(1, len(model_config["user_k"])):
            user_layers.append(torch.nn.Linear(model_config["user_k"][k-1], model_config["user_k"][k], device=self.device))
        self.user_linear_layers = torch.nn.ModuleList(user_layers)

        item_layers = [torch.nn.Linear(dim1item, model_config["item_k"][0], device=self.device)]
        for k in range(1, len(model_config["item_k"])):
            item_layers.append(torch.nn.Linear(model_config["item_k"][k - 1], model_config["item_k"][k], device=self.device))
        self.item_linear_layers = torch.nn.ModuleList(item_layers)

    def log(self, batch, which):
        if self.append_CF_after_ffn:
            raise NotImplementedError("implement this part if needed, or know what is logged exactly")

        input_ids = batch['input_ids']
        att_mask = batch['attention_mask']
        output = self.bert.forward(input_ids=input_ids,
                                   attention_mask=att_mask)
        if self.agg_strategy == "CLS":
            bert_rep = output.pooler_output
        elif self.agg_strategy == "mean_last":
            tokens_embeddings = output.last_hidden_state
            mask = att_mask.unsqueeze(-1).expand(tokens_embeddings.size()).float()
            tokens_embeddings = tokens_embeddings * mask
            sum_tokons = torch.sum(tokens_embeddings, dim=1)
            summed_mask = torch.clamp(att_mask.sum(1).type(torch.float), min=1e-9)
            bert_rep = (sum_tokons.T / summed_mask).T # divide by how many tokens (1s) are in the att_mask

        if which == "user":
            ffn_rep = bert_rep.copy()
            for k in range(len(self.user_linear_layers)-1):
                ffn_rep = torch.nn.functional.relu(self.user_linear_layers[k](ffn_rep))
            ffn_rep = self.user_linear_layers[-1](ffn_rep)
        elif which == "item":
            ffn_rep = bert_rep.copy()
            for k in range(len(self.item_linear_layers) - 1):
                ffn_rep = torch.nn.functional.relu(self.item_linear_layers[k](ffn_rep))
            ffn_rep = self.item_linear_layers[-1](ffn_rep)
        else:
            raise ValueError("user or item?")
        return bert_rep, ffn_rep

    # only works for the non-rl pipeline
    def prec_representations_for_test(self, users, items, padding_token):
        collate_fn = CollateUserItem(padding_token=padding_token)

        # user:
        dataloader = DataLoader(users, batch_size=1024, collate_fn=collate_fn)
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        reps = []
        for batch_idx, batch in pbar:
            user_ids = batch[INTERNAL_USER_ID_FIELD]
            input_ids = batch["input_ids"].to(self.device)
            att_mask = batch["attention_mask"].to(self.device)
            output = self.bert.forward(input_ids=input_ids,
                                       attention_mask=att_mask)
            if self.agg_strategy == "CLS":
                rep = output.pooler_output
            elif self.agg_strategy == "mean_last":
                tokens_embeddings = output.last_hidden_state
                mask = att_mask.unsqueeze(-1).expand(tokens_embeddings.size()).float()
                tokens_embeddings = tokens_embeddings * mask
                sum_tokons = torch.sum(tokens_embeddings, dim=1)
                summed_mask = torch.clamp(att_mask.sum(1).type(torch.float), min=1e-9)
                rep = (sum_tokons.T / summed_mask).T  # divide by how many tokens (1s) are in the att_mask

            for k in range(len(self.user_linear_layers) - 1):
                rep = torch.nn.functional.relu(self.user_linear_layers[k](rep))
            rep = self.user_linear_layers[-1](rep)

            if self.append_cf_after_ffn:
                rep = torch.cat([rep, self.user_embedding_CF(user_ids)], dim=1)

            reps.extend(rep.tolist())
        self.user_prec_reps = torch.nn.Embedding.from_pretrained(torch.tensor(reps)).to(self.device)

        #item:
        dataloader = DataLoader(items, batch_size=512, collate_fn=collate_fn)
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        reps = []
        for batch_idx, batch in pbar:
            item_ids = batch[INTERNAL_ITEM_ID_FIELD]
            input_ids = batch["input_ids"].to(self.device)
            att_mask = batch["attention_mask"].to(self.device)
            output = self.bert.forward(input_ids=input_ids,
                                       attention_mask=att_mask)
            if self.agg_strategy == "CLS":
                rep = output.pooler_output
            elif self.agg_strategy == "mean_last":
                tokens_embeddings = output.last_hidden_state
                mask = att_mask.unsqueeze(-1).expand(tokens_embeddings.size()).float()
                tokens_embeddings = tokens_embeddings * mask
                sum_tokons = torch.sum(tokens_embeddings, dim=1)
                summed_mask = torch.clamp(att_mask.sum(1).type(torch.float), min=1e-9)
                rep = (sum_tokons.T / summed_mask).T  # divide by how many tokens (1s) are in the att_mask

            for k in range(len(self.item_linear_layers) - 1):
                rep = torch.nn.functional.relu(self.item_linear_layers[k](rep))
            rep = self.item_linear_layers[-1](rep)

            if self.append_cf_after_ffn:
                rep = torch.cat([rep, self.item_embedding_CF(item_ids)], dim=1)

            reps.extend(rep.tolist())
        self.item_prec_reps = torch.nn.Embedding.from_pretrained(torch.tensor(reps)).to(self.device)

    def forward(self, batch, user_index=None, item_index=None, validate=False):
        # batch -> chunks * batch_size * tokens
        user_ids = batch[INTERNAL_USER_ID_FIELD].squeeze(1)
        item_ids = batch[INTERNAL_ITEM_ID_FIELD].squeeze(1)

        BERT_DIM = self.bert_embeddings.embedding_dim

        if (self.test_only or validate) and self.support_test_prec:
            user_rep = self.user_prec_reps(user_ids)
        else:
            input_ids = batch['user_input_ids']
            att_mask = batch['user_attention_mask']

            if user_index is not None:
                temp_user_ids = torch.Tensor([i for i, v in sorted(user_index.items(), key=lambda x: x[1])])
            else:
                temp_user_ids = user_ids

            output_u = self.bert.forward(input_ids=input_ids,
                                         attention_mask=att_mask)

            if self.agg_strategy == "CLS":
                user_rep = output_u.pooler_output
            elif self.agg_strategy == "mean_last":
                tokens_embeddings = output_u.last_hidden_state
                mask = att_mask.unsqueeze(-1).expand(tokens_embeddings.size()).float()
                tokens_embeddings = tokens_embeddings * mask
                sum_tokons = torch.sum(tokens_embeddings, dim=1)
                summed_mask = torch.clamp(att_mask.sum(1).type(torch.float), min=1e-9)
                user_rep = (sum_tokons.T / summed_mask).T # divide by how many tokens (1s) are in the att_mask

            for k in range(len(self.user_linear_layers) - 1):
                user_rep = torch.nn.functional.relu(self.user_linear_layers[k](user_rep))
            user_rep = self.user_linear_layers[-1](user_rep)

            if self.append_cf_after_ffn:
                user_rep = torch.cat([user_rep, self.user_embedding_CF(temp_user_ids)], dim=1)

            # repeat the user-rep by the batch occurrence
            if user_index is not None:
                repeats = []
                for id in user_ids:
                    repeats.append(user_index[id.item()])
                user_rep = torch.cat([user_rep[id, :].unsqueeze(0) for id in repeats], dim=0).to(self.device)

        if (self.test_only or validate) and self.support_test_prec:
            item_rep = self.item_prec_reps(item_ids)
        else:
            input_ids = batch['item_input_ids']
            att_mask = batch['item_attention_mask']

            if item_index is not None:
                temp_item_ids = torch.Tensor([i for i, v in sorted(item_index.items(), key=lambda x: x[1])])
            else:
                temp_item_ids = item_ids

            output_i = self.bert.forward(input_ids=input_ids,
                                         attention_mask=att_mask)
            if self.agg_strategy == "CLS":
                item_rep = output_i.pooler_output
            elif self.agg_strategy == "mean_last":
                tokens_embeddings = output_i.last_hidden_state
                mask = att_mask.unsqueeze(-1).expand(tokens_embeddings.size()).float()
                tokens_embeddings = tokens_embeddings * mask
                sum_tokons = torch.sum(tokens_embeddings, dim=1)
                summed_mask = torch.clamp(att_mask.sum(1).type(torch.float), min=1e-9)  #-> I see the point, but it's better to leave it as is to find the errors as in our case there should be something
                item_rep = (sum_tokons.T / summed_mask).T # divide by how many tokens (1s) are in the att_mask

            for k in range(len(self.item_linear_layers) - 1):
                item_rep = torch.nn.functional.relu(self.item_linear_layers[k](item_rep))
            item_rep = self.item_linear_layers[-1](item_rep)

            if self.append_cf_after_ffn:
                item_rep = torch.cat([item_rep, self.item_embedding_CF(temp_item_ids)], dim=1)

             # repeat the item-rep by the batch occurrence
            if item_index is not None:
                repeats = []
                for id in item_ids:
                    repeats.append(item_index[id.item()])
                item_rep = torch.cat([item_rep[id, :].unsqueeze(0) for id in repeats], dim=0).to(self.device)

        result = torch.sum(torch.mul(user_rep, item_rep), dim=1)
        result = result.unsqueeze(1)
        return result  #, user_text  # user_text for debug  # do not apply sigmoid here, later in the trainer if we wanted we would

