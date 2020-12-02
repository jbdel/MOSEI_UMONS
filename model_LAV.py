import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.fc import MLP
from layers.layer_norm import LayerNorm

# ------------------------------------
# ---------- Masking sequence --------
# ------------------------------------
def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)

# ------------------------------
# ---------- Flattening --------
# ------------------------------


class AttFlat(nn.Module):
    def __init__(self, args, flat_glimpse, merge=False):
        super(AttFlat, self).__init__()
        self.args = args
        self.merge = merge
        self.flat_glimpse = flat_glimpse
        self.mlp = MLP(
            in_size=args.hidden_size,
            mid_size=args.ff_size,
            out_size=flat_glimpse,
            dropout_r=args.dropout_r,
            use_relu=True
        )

        if self.merge:
            self.linear_merge = nn.Linear(
                args.hidden_size * flat_glimpse,
                args.hidden_size * 2
            )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -1e9
            )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.flat_glimpse):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        if self.merge:
            x_atted = torch.cat(att_list, dim=1)
            x_atted = self.linear_merge(x_atted)

            return x_atted

        return torch.stack(att_list).transpose_(0, 1)

# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, args):
        super(SA, self).__init__()

        self.mhatt = MHAtt(args)
        self.ffn = FFN(args)

        self.dropout1 = nn.Dropout(args.dropout_r)
        self.norm1 = LayerNorm(args.hidden_size)

        self.dropout2 = nn.Dropout(args.dropout_r)
        self.norm2 = LayerNorm(args.hidden_size)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, args, shift=False):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(args, shift)
        self.mhatt2 = MHAtt(args, shift)
        self.ffn = FFN(args)

        self.dropout1 = nn.Dropout(args.dropout_r)
        self.norm1 = LayerNorm(args.hidden_size)

        self.dropout2 = nn.Dropout(args.dropout_r)
        self.norm2 = LayerNorm(args.hidden_size)

        self.dropout3 = nn.Dropout(args.dropout_r)
        self.norm3 = LayerNorm(args.hidden_size)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, mask=x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x

# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, args, shift=False):
        super(MHAtt, self).__init__()
        self.args = args
        self.shift = shift

        self.linear_v = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_k = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_q = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_merge = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_shift = nn.Linear(60, 60) # TODO: remove hardcoded numbers

        self.dropout = nn.Dropout(args.dropout_r)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

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

        atted = self.att(v, k, q, mask)

        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.args.hidden_size
        )
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        # query: 32x4x60x128
        # key^T: 32x4x128x60
        # attention: 32x4x60x60
        # value: 32x4x60x128

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        if not self.shift:
            att_map = F.softmax(scores, dim=-1)
            att_map = self.dropout(att_map)
        else:
            att_map = F.normalize(scores, dim=3)
            att_map = self.linear_shift(att_map)
            att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


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

# ---------------------------
# ---- FF + norm  -----------
# ---------------------------
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



class LAV_Block(nn.Module):
    def __init__(self, args, i, shift=False):
        super(LAV_Block, self).__init__()
        self.args = args
        self.sa1 = SA(args)
        self.sa2 = SGA(args, shift)
        self.sa3 = SGA(args, shift)

        self.last = (i == args.layer-1)
        if not self.last:
            self.att_lang = AttFlat(args, args.lang_seq_len, merge=False)
            self.att_audio = AttFlat(args, args.audio_seq_len, merge=False)
            self.att_vid = AttFlat(args, args.video_seq_len, merge=False)
            self.norm_l = LayerNorm(args.hidden_size)
            self.norm_a = LayerNorm(args.hidden_size)
            self.norm_v = LayerNorm(args.hidden_size)
            self.dropout = nn.Dropout(args.dropout_r)

    def forward(self, x, x_mask, y, y_mask, z, z_mask):

        ax = self.sa1(x, x_mask)
        ay = self.sa2(y, x, y_mask, x_mask)
        az = self.sa3(z, x, z_mask, x_mask)

        x = ax + x
        y = ay + y
        z = az + z

        if self.last:
            return x, y, z

        ax = self.att_lang(x, x_mask)
        ay = self.att_audio(y, y_mask)
        az = self.att_vid(z, z_mask)

        return self.norm_l(x + self.dropout(ax)), \
               self.norm_a(y + self.dropout(ay)), \
               self.norm_v(z + self.dropout(az))



class Model_LAV(nn.Module):
    def __init__(self, args, vocab_size, pretrained_emb, shift=False):
        super(Model_LAV, self).__init__()

        self.args = args

        # LSTM
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

        # Feature size to hid size
        self.adapter_y = nn.Linear(args.audio_feat_size, args.hidden_size)
        self.adapter_z = nn.Linear(args.video_feat_size, args.hidden_size)

        # Encoder blocks
        self.enc_list = nn.ModuleList(
            [LAV_Block(args, i, shift) for i in range(args.layer)])

        # Flattenting features before proj
        self.attflat_ac   = AttFlat(args, 1, merge=True)
        self.attflat_vid  = AttFlat(args, 1, merge=True)
        self.attflat_lang = AttFlat(args, 1, merge=True)

        # Classification layers
        self.proj_norm = LayerNorm(2 * args.hidden_size)
        if self.args.task == "sentiment":
            if self.args.task_binary:
                self.proj = nn.Linear(2 * args.hidden_size, 2)
            else:
                self.proj = nn.Linear(2 * args.hidden_size, 7)
        if self.args.task == "emotion":
            self.proj = self.proj = nn.Linear(2 * args.hidden_size, 6)

    def forward(self, x, y, z):
        x_mask = make_mask(x.unsqueeze(2))
        y_mask = make_mask(y)
        z_mask = make_mask(z)


        embedding = self.embedding(x)

        x, _ = self.lstm_x(embedding)
        # y, _ = self.lstm_y(y)

        y, z = self.adapter_y(y), self.adapter_z(z)

        for i, dec in enumerate(self.enc_list):
            x_m, y_m, z_m = None, None, None
            if i == 0:
                x_m, y_m, z_m = x_mask, y_mask, z_mask
            x, y, z = dec(x, x_m, y, y_m, z, z_m)

        x = self.attflat_lang(
            x,
            None
        )

        y = self.attflat_ac(
            y,
            None
        )

        z = self.attflat_vid(
            z,
            None
        )


        # Classification layers
        proj_feat = x + y + z
        proj_feat = self.proj_norm(proj_feat)
        ans = self.proj(proj_feat)

        return ans
