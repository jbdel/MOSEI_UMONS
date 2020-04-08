import argparse
import torch
from torch.utils.data import DataLoader
from mosei_dataset import Mosei_Dataset
from net import MCA
from JB import JB
from glimpse import GLIMPSE
from glimpse2 import GLIMPSE2
from glimpse3 import GLIMPSE3
from model_bi import GLIMPSE4
from glimpse5 import GLIMPSE5
from glimpse6 import GLIMPSE6
from mono import MONO
import random

from train import evaluate
import glob
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='ckpt/')
    parser.add_argument('--name', type=str, default='exp0/')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)

    # Listing checkpoints
    ckpts = sorted(glob.glob(args.output + "/" + args.name +
                      '/best*'), reverse=True)

    # Loading original args
    args = torch.load(ckpts[0])['args']

    # Creating Test-set dataloader
    train_dset = Mosei_Dataset('train', args)
    test_dset = Mosei_Dataset('test', args, train_dset.token_to_ix)
    test_loader = DataLoader(test_dset, args.batch_size, num_workers=8, pin_memory=True)

    # Creating net
    net = eval(args.model)(args, train_dset.vocab_size, train_dset.pretrained_emb).cuda()

    # Ensembling
    ensemble_preds = {}
    ensemble_accuracies = []
    for ckpt in ckpts:

        # Getting current checkpoint predictions
        state_dict = torch.load(ckpt)['state_dict']
        net.load_state_dict(state_dict)
        test_accuracy, preds = evaluate(net, test_loader, args)
        for id, pred in preds.items():
            if id not in ensemble_preds:
                ensemble_preds[id] = []
            ensemble_preds[id].append(pred)
        print("Test accuracy for model " + ckpt + ":", test_accuracy)

        # Compute new ensembling accuracy
        ens_accuracy = []
        for step, (
                ids,
                _,
                _,
                ans,
        ) in enumerate(test_loader):
            ans = ans.cpu().data.numpy()
            for id, a in zip(ids, ans):
                avg = np.mean(np.array(ensemble_preds[id]), axis=0)
                if args.task == "sentiment":
                    ens_accuracy.append(np.argmax(avg) == np.argmax(a))
                if args.task == "emotion":
                    ens_accuracy.append((avg > 0) == a)
        print("New Ens. Accuracy :", 100 * np.mean(np.array(ens_accuracy)))
        ensemble_accuracies.append(100 * np.mean(np.array(ens_accuracy)))

    with open("best_scores", "a+") as f:
        f.write(str(max(ensemble_accuracies)) + " - " + args.output + "/" + args.name + "\n")
    print("max", str(max(ensemble_accuracies)))
