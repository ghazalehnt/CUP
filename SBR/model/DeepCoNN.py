from os.path import join

import gensim
import torch
import numpy as np


class CNN(torch.nn.Module):

    def __init__(self, config, word_dim):
        super(CNN, self).__init__()

        self.kernel_count = config["kernel_count"]

        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=word_dim,
                out_channels=config["kernel_count"],
                kernel_size=config["kernel_size"],
                padding=(config["kernel_size"] - 1) // 2),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool1d(1),  # out shape(batch_size,kernel_count,1)
            torch.nn.Dropout(p=config["dropout_prob"]))

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(config["kernel_count"], config["cnn_out_dim"]),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=config["dropout_prob"]))

    def forward(self, vec):
        latent = self.conv(vec.permute(0, 2, 1))
        latent = self.linear(latent.reshape(-1, self.kernel_count))
        return latent  # out shape(batch_size, cnn_out_dim)


class FactorizationMachine(torch.nn.Module):

    def __init__(self, p, k):  # p=cnn_out_dim
        super().__init__()
        self.v = torch.nn.Parameter(torch.rand(p, k) / 10)
        self.linear = torch.nn.Linear(p, 1, bias=True)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        linear_part = self.linear(x)  # input shape(batch_size, cnn_out_dim), out shape(batch_size, 1)
        inter_part1 = torch.mm(x, self.v) ** 2
        inter_part2 = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = torch.sum(inter_part1 - inter_part2, dim=1, keepdim=True)
        pair_interactions = self.dropout(pair_interactions)
        output = linear_part + 0.5 * pair_interactions
        return output  # out shape(batch_size, 1)


def load_embedding(word2vec_file, exp_dir):
    vocab = torch.load(join(exp_dir, "vocab.pth"))
    word_embedding = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file)
    weights = np.zeros((len(vocab), word_embedding.vectors.shape[1]))
    # wrt the saved vocab indexes:
    for v in word_embedding.index_to_key:
        weights[vocab[v]] = np.array(word_embedding.get_vector(v), dtype=np.float32)
    weights = torch.FloatTensor(weights)
    return weights, vocab


class DeepCoNN(torch.nn.Module):
    def __init__(self, config, exp_dir):
        super(DeepCoNN, self).__init__()
        weights, vocab = load_embedding(config['word2vec_file'], exp_dir)
        self.embedding = torch.nn.Embedding.from_pretrained(weights, padding_idx=vocab["<pad>"])
        self.cnn_u = CNN(config, word_dim=self.embedding.embedding_dim)
        self.cnn_i = CNN(config, word_dim=self.embedding.embedding_dim)
        self.fm = FactorizationMachine(config["cnn_out_dim"] * 2, 10)
        self.max_tokens = config["max_tokens"] if "max_tokens" in config else None

    def forward(self, batch):
        if self.max_tokens is not None:
            user_review = batch["user_tokenized_text"][:, :self.max_tokens]
            item_review = batch["item_tokenized_text"][:, :self.max_tokens]
        else:
            user_review = batch["user_tokenized_text"]
            item_review = batch["item_tokenized_text"]

        u_vec = self.embedding(user_review)
        i_vec = self.embedding(item_review)

        user_latent = self.cnn_u(u_vec)
        item_latent = self.cnn_i(i_vec)

        concat_latent = torch.cat((user_latent, item_latent), dim=1)
        prediction = self.fm(concat_latent)
        return prediction

