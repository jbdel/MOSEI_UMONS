import torch.nn as nn

from layers.fc import MLP
from layers.layer_norm import LayerNorm

import torch.nn as nn
import torch.nn.functional as F
from net import SA, SGA
import torch
import math
from layers.fc import FC

# Masking the sequence mask
def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)


# ------------------------------
# ---------- Flattening --------
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, args):
        super(AttFlat, self).__init__()
        self.args = args

        self.mlp = MLP(
            in_size=args.hidden_size,
            mid_size=args.flat_mlp_size,
            out_size=args.flat_glimpses,
            dropout_r=args.dropout_r,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            args.hidden_size * args.flat_glimpses,
            args.hidden_size * 2
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.args.flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, args):
        super(MHAtt, self).__init__()
        self.args = args

        self.linear_k = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_q = nn.Linear(args.hidden_size, args.hidden_size)

    def forward(self, k, q):
        n_batches = q.size(0)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        d_k = q.size(-1)

        scores = torch.matmul(
            q, k.transpose(-2, -1)
        ) / math.sqrt(d_k)

        return scores


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, args):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=args.hidden_size,
            mid_size=args.ff_size,
            out_size=args.hidden_size,
            dropout_r=args.dropout_r,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)



# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------



class SAC(nn.Module):
    def __init__(self, args):
        super(SAC, self).__init__()

        self.args = args
        self.mhatt = MHAtt(args)
        self.dropout = nn.Dropout(args.dropout_r)
        self.linear_vx = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_mergex = nn.Linear(args.hidden_size, args.hidden_size)

        self.linear_vy = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_mergey = nn.Linear(args.hidden_size, args.hidden_size)

    def forward(self, x, y, x_mask=None, y_mask=None):
        n_batches = x.size(0)

        scores = self.mhatt(x, y)


        #MASKING
        scoresx = scores
        if x_mask is not None:
            scoresx = scores.masked_fill(x_mask, -1e9)
        att_mapx = F.softmax(scoresx, dim=-1)
        att_mapx = self.dropout(att_mapx)

        #ATTENDING
        attedx = torch.matmul(att_mapx, self.linear_vx(x).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2))
        attedx = attedx.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.args.hidden_size
        )
        mergex = self.linear_mergex(attedx)

        # QV^T = (VQ^T)T
        scoresy = scores.transpose(-2, -1)
        if y_mask is not None:
            scoresy = scoresy.masked_fill(y_mask, -1e9)
        att_mapy = F.softmax(scoresy, dim=-1)
        att_mapy = self.dropout(att_mapy)
        attedy = torch.matmul(att_mapy, self.linear_vy(y).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2))
        attedy = attedy.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.args.hidden_size
        )
        mergey = self.linear_mergey(attedy)
        # residual
        y = self.dropout(mergex) + y
        x = self.dropout(mergey) + x

        return x, y

class Att_solo(nn.Module):
    def __init__(self, args, FLAT_GLIMPSES):
        super(Att_solo, self).__init__()
        self.args = args
        self.linear = nn.Linear(args.ff_size, FLAT_GLIMPSES)
        self.FLAT_GLIMPSES = FLAT_GLIMPSES

    def forward(self, x, q, x_mask=None):
        att = self.linear(q)
        if x_mask is not None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -1e9
            )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        return torch.stack(att_list).transpose_(0, 1)


class FFAndNorm(nn.Module):
    def __init__(self, args):
        super(FFAndNorm, self).__init__()

        self.ffn = FFN(args)
        self.norm1 = LayerNorm(args.hidden_size)
        self.dropout2 = nn.Dropout(args.dropout_r)
        self.norm2 = LayerNorm(args.hidden_size)

    def forward(self, x):
        x = self.norm1(x)
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x



class Block(nn.Module):
    def __init__(self, args):
        super(Block, self).__init__()
        self.args = args
        self.sa1 = SA(args)
        self.sa3 = SGA(args)
        self.fc1 = FC(args.hidden_size, args.ff_size, dropout_r=args.dropout_r, use_relu=True)
        self.fc2 = FC(args.hidden_size, args.ff_size, dropout_r=args.dropout_r, use_relu=True)
        self.att_lang = Att_solo(args, args.lang_seq_len)
        self.att_audio = Att_solo(args, args.audio_seq_len)

    def forward(self, x, x_mask, y, y_mask):

        ax = self.sa1(x, x_mask)
        ay = self.sa3(y, x, y_mask, x_mask)

        x = ax + x
        y = ay + y

        ax = self.att_lang(x, self.fc1(x), x_mask)
        ay = self.att_audio(y, self.fc2(y), y_mask)

        return ax + x, ay + y


class Model_bi(nn.Module):
    def __init__(self, args, vocab_size, pretrained_emb):
        super(Model_bi, self).__init__()

        self.args = args

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=args.word_embed_size
        )

        # Loading the GloVe embedding weights
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm_x = nn.LSTM(
            input_size=args.word_embed_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            batch_first=True
        )

        # self.lstm_y = nn.LSTM(
        #     input_size=args.audio_feat_size,
        #     hidden_size=args.hidden_size,
        #     num_layers=1,
        #     batch_first=True
        # )


        self.adapter = nn.Linear(args.audio_feat_size, args.hidden_size)

        self.dec_list = nn.ModuleList([Block(args) for _ in range(args.layer)])

        #flattening
        self.attflat_img = AttFlat(args)
        self.attflat_lang = AttFlat(args)

        # Classification layers
        self.proj_norm = LayerNorm(2 * args.hidden_size)

        if self.args.task == "sentiment":
            if self.args.task_binary:
                self.proj = nn.Linear(2 * args.hidden_size, 2)
            else:
                self.proj = nn.Linear(2 * args.hidden_size, 7)
        if self.args.task == "emotion":
            self.proj = self.proj = nn.Linear(2 * args.hidden_size, 6)

    def forward(self, x, y):
        x_mask = make_mask(x.unsqueeze(2))
        y_mask = make_mask(y)

        embedding = self.embedding(x)

        x, _ = self.lstm_x(embedding)
        # y, _ = self.lstm_y(y)

        y = self.adapter(y)

        for i, dec in enumerate(self.dec_list):
            x_m, x_y = None, None
            if i == 0:
                x_m, x_y = x_mask, y_mask
            x, y = dec(x, x_m, y, x_y)

        # Flatten to vector #enlever ????
        x = self.attflat_lang(
            x,
            x_mask
        )

        y = self.attflat_img(
            y,
            y_mask
        )

        # Classification layers
        proj_feat = x + y
        proj_feat = self.proj_norm(proj_feat)
        ans = self.proj(proj_feat)

        return ans



