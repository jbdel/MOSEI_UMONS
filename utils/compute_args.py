import torch
import numpy as np



def compute_args(args):

    # DataLoader
    if args.dataset == "MOSEI": args.dataloader = 'Mosei_Dataset'
    if args.dataset == "MELD": args.dataloader = 'Meld_Dataset'

    # Loss function to use
    if args.dataset == 'MOSEI' and args.task == 'sentiment': args.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    if args.dataset == 'MOSEI' and args.task == 'emotion': args.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    if args.dataset == 'MELD': args.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    # Answer size
    if args.dataset == 'MOSEI' and args.task == "sentiment": args.ans_size = 7
    if args.dataset == 'MOSEI' and args.task == "sentiment" and args.task_binary: args.ans_size = 2
    if args.dataset == 'MOSEI' and args.task == "emotion": args.ans_size = 6
    if args.dataset == 'MELD' and args.task == "emotion": args.ans_size = 7
    if args.dataset == 'MELD' and args.task == "sentiment": args.ans_size = 3


    if args.dataset == 'MOSEI': args.pred_func = "amax"
    if args.dataset == 'MOSEI' and args.task == "emotion": args.pred_func = "multi_label"
    if args.dataset == 'MELD': args.pred_func = "amax"

    return args