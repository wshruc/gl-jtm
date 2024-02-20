import sys
import torch as th
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import warnings
import numpy as np
import math
from layers import MultiHeadedAttention, ScaledDotProductAttention
from blitz.modules import BayesianGRU, BayesianConv1d, BayesianLinear, BayesianLSTM

warnings.filterwarnings('ignore')


class JtBNN(nn.Module):
    def __init__(self, dropout, learning_rate, margin, hidden_size, word_emb, rel_emb, rel_dict):
        super(JtBNN, self).__init__()
        self.word_embedding = nn.Embedding(word_emb.shape[0], word_emb.shape[1])
        self.word_embedding.weight = nn.Parameter(th.from_numpy(word_emb).float())
        self.word_embedding.weight.requires_grad = False

        self.rel_embedding = nn.Embedding(rel_emb.shape[0], rel_emb.shape[1])
        self.rel_embedding.weight = nn.Parameter(th.from_numpy(rel_emb).float())
        self.rel_embedding.weight.requires_grad = False  # fix the embedding matrix

        self.rel_dict = rel_dict
        self.dropout = True  # dropout 0.35
        self.dropout_rate = dropout
        self.embedding_dim = word_emb.shape[1]  # 300
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate  # 0.001
        self.margin = margin  # 0.5
        self.window_size = [2, 3, 4, 5]
        self.re_len = 20 #sq20 # Different datasets have different relation tokens length, For wq self.re_len = 20, for sq self.re_len = 16
        self.hidden_er = 150  # 150

        # Multi-Head Attention
        self.mlhatt = MultiHeadedAttention(4, self.embedding_dim, dropout=self.dropout_rate)
        # Attention
        self.sdpatt = ScaledDotProductAttention()

        if self.dropout:
            self.rnn_dropout = nn.Dropout(p=self.dropout_rate)

        self.first_gru = BayesianGRU(self.embedding_dim, self.hidden_size)
        self.second_gru = BayesianGRU(self.embedding_dim, self.hidden_size)

        self.convs = nn.ModuleList([nn.Sequential(
            BayesianConv1d(in_channels=300, out_channels=150, kernel_size=h),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.re_len - h + 1))
            for h in self.window_size
        ])
        self.proj_rel = nn.Sequential(nn.Linear(4 * self.hidden_size, self.hidden_size), \
                                      nn.ReLU())

        self.fc = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.hidden_size * 2, self.hidden_size * 1),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.hidden_size * 1, 1)
        )

        """
         Note:
        For SimpleQuestions, when set self.hidden_er to 300, we can obtain the SOTA result
        For WebQuestions, when set self.hidden_er to 150, we can obtain the SOTA result
        """
        self.hidden2tag = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_er),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.hidden_er, 4)
        )
        self.proj1 = nn.Linear(5 * self.hidden_size, self.hidden_size * 3)
        self.proj2 = nn.Linear(self.hidden_size * 3, self.hidden_size * 1)
        self.gate_ent = nn.Linear(self.hidden_size, self.hidden_size)
        self.gate_rel = nn.Linear(self.hidden_size, self.hidden_size)#BayesianConv1d(in_channels=300, out_channels=300, kernel_size=3, padding=1)
    def apply_multiple(self, x):
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        return th.cat([p1, p2], 1)

    def soft_attention_align(self, x1, x2, mask1, mask2):
        '''
        x1: batch_size * seq_len * dim
        x2: batch_size * seq_len * dim
        '''
        # attention: batch_size * seq_len * seq_len
        attention = th.matmul(x1, x2.transpose(1, 2))
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))

        # weight: batch_size * seq_len * seq_len
        weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)
        x1_align = th.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        x2_align = th.matmul(weight2, x1)
        # x_align: batch_size * seq_len * hidden_size
        return x1_align, x2_align, weight1, weight2

    def submul(self, x1, x2):
        mul = x1 * x2
        sub = th.abs(x1 - x2)
        sum = x1 + x2
        return th.cat([sub, mul, sum], -1)

    def forward(self, question, word_relation, rel_relation):
        q_mask = question.eq(0)#.unsqueeze(2)
        w_mask = word_relation.eq(0)#.unsqueeze(2)
        r_mask = rel_relation.eq(0)
        """
        Embedding layer
        """
        q_embed = self.word_embedding(question)
        w_embed = self.word_embedding(word_relation)
        """
        KG embedding
        """
        r_embed = self.rel_embedding(rel_relation)
        if self.dropout:
            q_embed = self.rnn_dropout(q_embed)
            w_embed = self.rnn_dropout(w_embed)
            r_embed = self.rnn_dropout(r_embed)

        rel_embed = th.cat([w_embed, r_embed], dim=1)
        rel_mask = th.cat([w_mask, r_mask], dim=1)

        """
        Bayesian GRU1
        """
        q_encoded, q_hidden = self.first_gru(q_embed)
        w_encoded, w_hidden = self.first_gru(w_embed)

        # entity detection
        q_aligned, w_aligned, weight1, weight2 = self.soft_attention_align(q_encoded, w_encoded, q_mask, w_mask)

        q_combined = th.cat([q_encoded, q_aligned, self.submul(q_encoded, q_aligned)], dim=-1)
        w_combined = th.cat([w_encoded, w_aligned, self.submul(w_encoded, w_aligned)], dim=-1)

        # projected_q = self.proj_rel(q_combined)
        # projected_w = self.proj_rel(w_combined)
        projected_q = self.proj2(F.relu(self.proj1(q_combined)))
        projected_w = self.proj2(F.relu(self.proj1(w_combined)))
        #
        if self.dropout:
            projected_q = self.rnn_dropout(projected_q)
            projected_w = self.rnn_dropout(projected_w)
        """
        将上步结果输入Bayesian GRU 2
        """
        qr_merge, w_q_weight = self.sdpatt(projected_q, projected_w, projected_w, q_mask.unsqueeze(2))

        ent_inf = th.tanh(self.gate_ent(qr_merge)) * qr_merge #+ qr_merge
        # entity recognition
        q_compare, _ = self.second_gru(ent_inf)
        e_scores = self.hidden2tag(q_compare)

        rel_inf = th.tanh(self.gate_rel(qr_merge)) * qr_merge
        qr_mergess = rel_inf + q_compare
        c_r_input = qr_mergess.permute(0, 2, 1)
        c_r_output = [conv(c_r_input) for conv in self.convs]
        c_r_output = th.cat(c_r_output, dim=1)
        c_r_output = c_r_output.view(-1, c_r_output.size(1))

        w_rep = self.apply_multiple(qr_mergess)

        merged = th.cat([c_r_output, w_rep], dim=1)
        score = self.fc(merged)
        """
        经过sigmoid激活函数
        """
        score = F.sigmoid(score)
        return e_scores, score, w_q_weight


    def loss_function(self, logits, target, masks, device, num_class=4):
        criterion = nn.CrossEntropyLoss(reduction='none')
        logits = logits.view(-1, num_class)
        target = target.view(-1)
        masks = masks.view(-1)
        cross_entropy = criterion(logits, target)
        loss = cross_entropy * masks
        loss = loss.sum() / (masks.sum() + 1e-12)  # 加上 1e-12 防止被除数为 0
        loss = loss.to(device)
        return loss

if __name__ == '__main__':
    pass
