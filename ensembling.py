import argparse, os, glob, pickle, warnings, torch
import numpy as np
from utils.pred_func import *
from sklearn.metrics import classification_report
from utils.compute_args import compute_args
from torch.utils.data import DataLoader
from mosei_dataset import Mosei_Dataset
from meld_dataset import Meld_Dataset
from model_LA import Model_LA
from model_LAV import Model_LAV
from train import evaluate
warnings.filterwarnings("ignore")


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='ckpt/')
    parser.add_argument('--name', type=str, default='exp0/')
    parser.add_argument('--sets',  nargs='+', default=["valid", "test"])

    parser.add_argument('--index', type=int, default=99)
    parser.add_argument('--private_set', type=str, default=None)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Save vars
    private_set = args.private_set
    index = args.index
    sets = args.sets

    # Listing sorted checkpoints
    ckpts = sorted(glob.glob(os.path.join(args.output, args.name,'best*')), reverse=True)

    # Load original args
    args = torch.load(ckpts[0])['args']
    args = compute_args(args)


    # Define the splits to be evaluated
    evaluation_sets = list(sets) + ([private_set] if private_set is not None else [])
    print("Evaluated sets: ", str(evaluation_sets))
    # Creating dataloader
    train_dset = eval(args.dataloader)('train', args)
    loaders = {set: DataLoader(eval(args.dataloader)(set, args, train_dset.token_to_ix),
               args.batch_size,
               num_workers=8,
               pin_memory=True) for set in evaluation_sets}

    # Creating net
    net = eval(args.model)(args, train_dset.vocab_size, train_dset.pretrained_emb).cuda()

    # Ensembling sets
    ensemble_preds = {set: {} for set in evaluation_sets}
    ensemble_accuracies = {set: [] for set in evaluation_sets}

    # Iterating over checkpoints
    for i, ckpt in enumerate(ckpts):

        if i >= index:
            break

        print("###### Ensembling " + str(i+1))
        state_dict = torch.load(ckpt)['state_dict']
        net.load_state_dict(state_dict)

        # Evaluation per checkpoint predictions
        for set in evaluation_sets:
            accuracy, preds = evaluate(net, loaders[set], args)
            print('Accuracy for ' + set + ' for model ' + ckpt + ":", accuracy)
            for id, pred in preds.items():
                if id not in ensemble_preds[set]:
                    ensemble_preds[set][id] = []
                ensemble_preds[set][id].append(pred)

            # Compute set ensembling accuracy
            # Get all ids and answers
            ids = [id for ids, _, _, _, _ in loaders[set] for id in ids]
            ans = [np.array(a) for _, _, _, _, ans in loaders[set] for a in ans]

            # for all id, get averaged probabilities
            avg_preds = np.array([np.mean(np.array(ensemble_preds[set][id]), axis=0) for id in ids])
            # Compute accuracies
            if set != private_set:
                accuracy = np.mean(eval(args.pred_func)(avg_preds) == ans) * 100
                print("New " + set + " ens. Accuracy :", accuracy)
                ensemble_accuracies[set].append(accuracy)

            if i + 1 == index:
                print(classification_report(ans, eval(args.pred_func)(avg_preds)))

    # Printing overall results
    for set in sets:
        print("Max ensemble w-accuracies for " + set + " : " + str(max(ensemble_accuracies[set])))

