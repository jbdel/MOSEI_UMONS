from __future__ import print_function
import os
import pickle
import numpy as np
import torch
from utils.plot import plot
from utils.tokenize import tokenize, create_dict, sent_to_ix, cmumosei_2, cmumosei_7, pad_feature

from torch.utils.data import Dataset

class Mosei_Dataset(Dataset):
    def __init__(self, name, args, token_to_ix=None, dataroot='data'):
        super(Mosei_Dataset, self).__init__()
        self.args = args
        assert name in ['train', 'valid', 'test']

        id_file = os.path.join(dataroot, "d_"+name+".pkl")
        word_file = os.path.join(dataroot, "w_"+name+".pkl")
        # audio_file = os.path.join(dataroot, "mel_"+name+"_short.p")
        audio_file = os.path.join(dataroot, "mag_"+name+"_r_16.p")
        # audio_file = os.path.join(dataroot, "mel_"+name+".pkl")
        sy_file = os.path.join(dataroot, "sy_"+name+".pkl")
        ey_file = os.path.join(dataroot, "ey_"+name+".pkl")

        self.index_to_video_id = pickle.load(open(id_file, "rb"))
        self.index_to_word = pickle.load(open(word_file, "rb"))
        self.index_to_audio = pickle.load(open(audio_file, "rb"))
        if args.task == "emotion":
            self.index_to_label = pickle.load(open(ey_file, "rb"))
        if args.task == "sentiment":
            self.index_to_label = pickle.load(open(sy_file, "rb"))


        for key in self.index_to_video_id.keys():
            assert self.index_to_video_id[key][0] == \
                   self.index_to_word[key][0] == \
                   self.index_to_label[key][0], "nope"


        # big = []
        # for key in self.index_to_audio.keys():
        #     x = self.index_to_audio[key].shape[0]
        #     big.append(x)
        # plot(big)
        # sys.exit()

        # big = []
        # for k in self.index_to_audio:
        #     x = k.shape[0]
        #     big.append(x)
        # plot(big)
        # sys.exit()

        # create list of sentences
        self.sentence_list = tokenize(self.index_to_word)
        # create dictionary and glove embeddings
        if token_to_ix is not None:
            self.token_to_ix = token_to_ix
        else: # Train
            self.token_to_ix, self.pretrained_emb = create_dict(self.sentence_list, dataroot)

        self.vocab_size = len(self.token_to_ix)
        self.l_max_len = args.lang_seq_len
        self.a_max_len = args.audio_seq_len

    def __getitem__(self, idx):
        # l = self.pad_feature(self.index_to_word[index][1], self.max_len)

        #video_id
        id = self.index_to_video_id[idx][0]
        # language
        l = sent_to_ix(self.sentence_list[idx], self.token_to_ix, max_token=self.l_max_len)
        # audio
        a = pad_feature(self.index_to_audio[idx], self.a_max_len)
        #label
        label = self.index_to_label[idx][1]
        if self.args.task == "sentiment" and self.args.task_binary:
            c = cmumosei_2(label)
            y = np.zeros(2, np.float32)
            y[c] = 1
        if self.args.task == "sentiment" and not self.args.task_binary:
            c = cmumosei_7(label)
            y = np.zeros(7, np.float32)
            y[c] = 1
        if self.args.task == "emotion":
            label[label > 0] = 1
            y = label

        return id, torch.from_numpy(l), torch.from_numpy(a), torch.from_numpy(y)

    def __len__(self):
        return len(self.index_to_video_id)