"""DuoRec model for RecBole-style sequential recommendation.

Adapted from the official DuoRec implementation and adjusted to run on
RecBole 1.2.x pipelines in this repository.
"""

import random

import numpy as np
import torch
from torch import nn

from recbole.data.interaction import Interaction
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


class DuoRec(SequentialRecommender):
    """DuoRec with in-model semantic positive fallback.

    The original DuoRec code expected semantic augmentations to be injected by a
    dedicated dataloader. Our pipeline does not provide those extra fields, so
    we generate them on the fly when needed.
    """

    input_type = "point"

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.dataset = dataset

        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.lmd = config["lmd"]
        self.lmd_sem = config["lmd_sem"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        self.ssl = config["contrast"]
        self.tau = config["tau"]
        self.sim = config["sim"]
        self.batch_size = config["train_batch_size"]
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.semantic_sample_max_tries = int(config["semantic_sample_max_tries"]) if "semantic_sample_max_tries" in config else 3
        self.aug_nce_fct = nn.CrossEntropyLoss()
        self.sem_aug_nce_fct = nn.CrossEntropyLoss()

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.dataset_item_seq = dataset.inter_feat[self.ITEM_SEQ]
        self.dataset_item_seq_len = dataset.inter_feat[self.ITEM_SEQ_LEN]
        self.same_item_index = self._build_same_item_index(dataset)
        self.apply(self._init_weights)

    def _build_same_item_index(self, dataset):
        same_target_index = {}
        target_item = dataset.inter_feat[self.ITEM_ID]
        target_item_np = target_item.cpu().numpy() if torch.is_tensor(target_item) else np.asarray(target_item)
        for idx, item_id in enumerate(target_item_np.tolist()):
            key = int(item_id)
            if key not in same_target_index:
                same_target_index[key] = []
            same_target_index[key].append(int(idx))
        return {key: np.asarray(val, dtype=np.int64) for key, val in same_target_index.items()}

    def _sample_semantic_positive_batch(self, interaction):
        sem_pos_lengths = []
        sem_pos_seqs = []

        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        target_items = interaction[self.ITEM_ID]
        item_seq_cpu = item_seq.detach().cpu()
        item_seq_len_cpu = item_seq_len.detach().cpu()
        target_items_cpu = target_items.detach().cpu().tolist()

        for i, item_id in enumerate(target_items_cpu):
            item_id = int(item_id)
            candidates = self.same_item_index.get(item_id)
            if candidates is None or len(candidates) == 0:
                sem_pos_seqs.append(item_seq_cpu[i])
                sem_pos_lengths.append(item_seq_len_cpu[i])
                continue

            chosen_seq = None
            chosen_len = None
            if len(candidates) == 1:
                sample_index = int(candidates[0])
                chosen_seq = self.dataset_item_seq[sample_index]
                chosen_len = self.dataset_item_seq_len[sample_index]
            else:
                max_tries = max(1, int(self.semantic_sample_max_tries))
                for _ in range(max_tries):
                    sample_index = int(candidates[random.randrange(len(candidates))])
                    sample_item_list = self.dataset_item_seq[sample_index]
                    if not torch.equal(item_seq_cpu[i], sample_item_list):
                        chosen_seq = sample_item_list
                        chosen_len = self.dataset_item_seq_len[sample_index]
                        break
                if chosen_seq is None:
                    for sample_index in candidates:
                        sample_index = int(sample_index)
                        sample_item_list = self.dataset_item_seq[sample_index]
                        if not torch.equal(item_seq_cpu[i], sample_item_list):
                            chosen_seq = sample_item_list
                            chosen_len = self.dataset_item_seq_len[sample_index]
                            break

            if chosen_seq is None:
                chosen_seq = item_seq_cpu[i]
                chosen_len = item_seq_len_cpu[i]

            sem_pos_seqs.append(chosen_seq)
            if torch.is_tensor(chosen_len):
                sem_pos_lengths.append(chosen_len)
            else:
                sem_pos_lengths.append(torch.as_tensor(chosen_len))

        sem_pos_lengths = torch.stack(sem_pos_lengths).to(self.device, non_blocking=True)
        sem_pos_seqs = torch.stack(sem_pos_seqs).to(self.device, non_blocking=True)
        return sem_pos_seqs, sem_pos_lengths

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.layer_norm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output

    def calculate_loss(self, interaction):
        if self.ssl in {"us", "su", "us_x"}:
            if "sem_aug" not in interaction.interaction or "sem_aug_lengths" not in interaction.interaction:
                sem_aug, sem_aug_lengths = self._sample_semantic_positive_batch(interaction)
                interaction.update(
                    Interaction({"sem_aug": sem_aug, "sem_aug_lengths": sem_aug_lengths})
                )

        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        if self.ssl in {"us", "un"}:
            aug_seq_output = self.forward(item_seq, item_seq_len)
            nce_logits, nce_labels = self.info_nce(
                seq_output,
                aug_seq_output,
                temp=self.tau,
                batch_size=item_seq_len.shape[0],
                sim=self.sim,
            )
            loss += self.lmd * self.aug_nce_fct(nce_logits, nce_labels)

        if self.ssl in {"us", "su"}:
            sem_aug = interaction["sem_aug"]
            sem_aug_lengths = interaction["sem_aug_lengths"]
            sem_aug_seq_output = self.forward(sem_aug, sem_aug_lengths)
            sem_nce_logits, sem_nce_labels = self.info_nce(
                seq_output,
                sem_aug_seq_output,
                temp=self.tau,
                batch_size=item_seq_len.shape[0],
                sim=self.sim,
            )
            loss += self.lmd_sem * self.aug_nce_fct(sem_nce_logits, sem_nce_labels)

        if self.ssl == "us_x":
            aug_seq_output = self.forward(item_seq, item_seq_len)
            sem_aug = interaction["sem_aug"]
            sem_aug_lengths = interaction["sem_aug_lengths"]
            sem_aug_seq_output = self.forward(sem_aug, sem_aug_lengths)
            sem_nce_logits, sem_nce_labels = self.info_nce(
                aug_seq_output,
                sem_aug_seq_output,
                temp=self.tau,
                batch_size=item_seq_len.shape[0],
                sim=self.sim,
            )
            loss += self.lmd_sem * self.aug_nce_fct(sem_nce_logits, sem_nce_labels)

        return loss

    def mask_correlated_samples(self, batch_size):
        n = 2 * batch_size
        mask = torch.ones((n, n), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size, sim="dot"):
        n = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)

        if sim == "cos":
            sim_mat = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        else:
            sim_mat = torch.mm(z, z.T) / temp

        sim_i_j = torch.diag(sim_mat, batch_size)
        sim_j_i = torch.diag(sim_mat, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(n, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size).to(sim_mat.device)
        else:
            if self.mask_default.device != sim_mat.device:
                self.mask_default = self.mask_default.to(sim_mat.device)
            mask = self.mask_default
        negative_samples = sim_mat[mask].reshape(n, -1)

        labels = torch.zeros(n, device=positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        return torch.mul(seq_output, test_item_emb).sum(dim=1)

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        return torch.matmul(seq_output, test_items_emb.transpose(0, 1))
