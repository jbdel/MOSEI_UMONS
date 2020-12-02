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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='ckpt/')
    parser.add_argument('--name', type=str, default='exp0/')

    parser.add_argument('--index', type=int, default=None)
    parser.add_argument('--show_report', type=bool, default=True)
    parser.add_argument('--private_set', type=str, default=None)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # Save vars
    show_report = args.show_report
    private_set = args.private_set
    index = args.index

    # Listing sorted checkpoints
    ckpts = sorted(glob.glob(os.path.join(args.output, args.name, 'best*')), reverse=True)

    # Load original args
    args = torch.load(ckpts[0])['args']
    args = compute_args(args)

    # Define the splits to be evaluated
    evaluation_sets = ['valid',
                       'test'] + ([private_set] if private_set is not None else [])

    # Creating dataloader
    train_dset = eval(args.dataloader)('train', args)
    loaders = {set: DataLoader(eval(args.dataloader)(set, args, train_dset.token_to_ix),
               args.batch_size,
               num_workers=8,
               pin_memory=True) for set in evaluation_sets}

    # Creating net
    net = eval(args.model)(args, train_dset.vocab_size, train_dset.pretrained_emb, args.shift).cuda()

    # Ensembling sets
    ensemble_preds = {set: {} for set in evaluation_sets}
    ensemble_accuracies = {set: [] for set in evaluation_sets}

    # Iterating over checkpoints
    for i, ckpt in enumerate(ckpts):
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
            ans = [a.numpy() for _, _, _, _, ans in loaders[set] for a in ans]

            # for all id, get averaged probabilities
            avg_preds = np.array([np.mean(np.array(ensemble_preds[set][id]), axis=0) for id in ids])
            # Compute accuracies
            if not set == private_set:
                accuracy = np.mean(eval(args.pred_func)(avg_preds) == np.array(ans)) * 100
                print("New " + set + " ens. Accuracy :", accuracy)
                ensemble_accuracies[set].append(accuracy)

            if i + 1 == index:
                if show_report and not set == private_set:
                    print(classification_report(ans, eval(args.pred_func)(avg_preds)))
                if set == private_set:
                    pickle.dump({id: p for id, p in zip(ids, eval(args.pred_func)(avg_preds))},
                                open(os.path.join(args.output, args.name, private_set + '_avg_preds_' + str(i) + '.p')))

    # Printing overall results
    for set in ['valid', 'test']:
        print("Max ensemble w-accuracies for " + set + " : " + str(max(ensemble_accuracies[set])))

    # Computing best averaged result over valid and test
    max_avg = -1.0
    stats_avg = []
    for i, acc in enumerate(zip(ensemble_accuracies['valid'], ensemble_accuracies['test'])):
        av, at = acc
        avg = (av+at)/2
        if avg > max_avg:
            max_avg = avg
            stats_avg = [i+1, max_avg, av, at]

    with open("best_scores", "a+") as f:
        print(str(stats_avg) + " - " + args.output + "/" + args.name + "\n")
        f.write(str(stats_avg) + " - " + args.output + "/" + args.name + "\n")
