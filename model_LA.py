import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_LAV import make_mask, AttFlat, SA, SGA

from layers.fc import MLP
from layers.layer_norm import LayerNorm


class LA_Block(nn.Module):
    def __init__(self, args, i, shift=False):
        super(LA_Block, self).__init__()
        self.args = args
        self.sa1 = SA(args)
        self.sa3 = SGA(args, shift)

        self.last = (i == args.layer-1)
        if not self.last:
            self.att_lang = AttFlat(args, args.lang_seq_len, merge=False)
            self.att_audio = AttFlat(args, args.audio_seq_len, merge=False)
            self.norm_l = LayerNorm(args.hidden_size)
            self.norm_i = LayerNorm(args.hidden_size)
            self.dropout = nn.Dropout(args.dropout_r)

    def forward(self, x, x_mask, y, y_mask):

        ax = self.sa1(x, x_mask)
        ay = self.sa3(y, x, y_mask, x_mask)

        x = ax + x
        y = ay + y

        if self.last:
            return x, y

        ax = self.att_lang(x, x_mask)
        ay = self.att_audio(y, y_mask)

        return self.norm_l(x + self.dropout(ax)), \
               self.norm_i(y + self.dropout(ay))


class Model_LA(nn.Module):
    def __init__(self, args, vocab_size, pretrained_emb, shift=False):
        super(Model_LA, self).__init__()

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

        self.lstm_y = nn.LSTM(
            input_size=args.audio_feat_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Feature size to hid size
        # self.adapter = nn.Linear(args.audio_feat_size, args.hidden_size)

        # Encoder blocks
        self.enc_list = nn.ModuleList(
            [LA_Block(args, i, shift) for i in range(args.layer)])

        # Flattenting features before proj
        self.attflat_img  = AttFlat(args, 1, merge=True)
        self.attflat_lang = AttFlat(args, 1, merge=True)

        # Classification layers
        self.proj_norm = LayerNorm(2 * args.hidden_size)
        self.proj = self.proj = nn.Linear(2 * args.hidden_size, args.ans_size)

    def forward(self, x, y, _):
        x_mask = make_mask(x.unsqueeze(2))
        y_mask = make_mask(y)

        embedding = self.embedding(x)

        x, _ = self.lstm_x(embedding)
        y, _ = self.lstm_y(y)

        # y = self.adapter(y)

        for i, dec in enumerate(self.enc_list):
            x_m, x_y = None, None
            if i == 0:
                x_m, x_y = x_mask, y_mask
            x, y = dec(x, x_m, y, x_y)

        x = self.attflat_lang(
            x,
            None
        )

        y = self.attflat_img(
            y,
            None
        )

        # Classification layers
        proj_feat = x + y
        proj_feat = self.proj_norm(proj_feat)
        ans = self.proj(proj_feat)

        return ans
