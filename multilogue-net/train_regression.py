import numpy as np, torch, torch.nn as nn, torch.optim as optim
import argparse, time, pandas as pd, os
import sys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from model import RegressionModel, MaskedMSELoss, BiModalAttention
from dataloader import MOSEIRegression

np.random.seed(393)
torch.cuda.device([0])

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = range(size)
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_MOSEI_loaders(path, batch_size=128, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MOSEIRegression(path=path)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, collate_fn=trainset.collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler, collate_fn=trainset.collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    testset = MOSEIRegression(path=path, train=False)
    test_loader = DataLoader(testset, batch_size=batch_size, collate_fn=testset.collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):    
    losses, preds, labels, masks = [], [], [], []
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data] if cuda else data
        pred = model(textf, acouf, visuf, textf, qmask, umask) 
        labels_ = label.view(-1) 
        umask_ = umask.view(-1) 
        loss = loss_function(pred, labels_, umask_)
        preds.append(pred.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask_.cpu().numpy())
        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            optimizer.step()
    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), float('nan'), [], [], []
    avg_loss = round(np.sum(losses)/np.sum(masks),4)
    mae = round(mean_absolute_error(labels,preds,sample_weight=masks),4)
    pred_lab = pd.DataFrame(list(filter(lambda x: x[2]==1, zip(labels, preds, masks))))
    pear = round(pearsonr(pred_lab[0], pred_lab[1])[0], 4)
    return avg_loss, mae, pear, labels, preds, masks

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Trains a regression model for sentiment data with output ranging from -3 to +3 indicating sentiment")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.25, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=128, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--log_dir', type=str, default='logs/mosei_regression', help='Directory for tensorboard logs')
    parser.add_argument('--model_path', type=str, default='model/categorical.model', help='Path to the final model')
    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok = True)
    writer = SummaryWriter(args.log_dir)
    print(args)

    # Run on either GPU or CPU
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')
    print("Tensorboard logs in " + args.log_dir)

    batch_size = args.batch_size
    n_classes  = 6
    cuda       = args.cuda
    n_epochs   = args.epochs
    D_m_text, D_m_audio, D_m_video, D_m_context = 300, 384, 35, 300
    D_g, D_p, D_e, D_h, D_a = 150, 150, 100, 100, 100

    # Instantiate model
    model = RegressionModel(D_m_text, D_m_audio, D_m_video, D_m_context, D_g, D_p, D_e, D_h, dropout_rec=args.rec_dropout, dropout=args.dropout)

    if cuda:
        model.cuda()
    loss_function = MaskedMSELoss()

    # Get optimizer and relevant dataloaders
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    train_loader, valid_loader, test_loader = get_MOSEI_loaders('./data/regression.pkl', valid=0.0, batch_size=batch_size, num_workers=0)
    best_loss, best_label, best_pred, best_mask, best_pear = None, None, None, None, None

    # Training loop
    for e in tqdm(range(n_epochs), desc = 'MOSEI Regression'):
        train_loss, train_mae, train_pear,_,_,_ = train_or_eval_model(model, loss_function, train_loader, e, optimizer, True)
        test_loss, test_mae, test_pear, test_label, test_pred, test_mask = train_or_eval_model(model, loss_function, test_loader, e)
        writer.add_scalar("Train Loss - MOSEI Regression", train_loss, e)
        writer.add_scalar("Test Loss - MOSEI Regression", test_loss, e)
        writer.add_scalar("Train MAE - MOSEI Regression", train_mae, e)
        writer.add_scalar("Test MAE - MOSEI Regression", test_mae, e)
        writer.add_scalar("Train Pearson - MOSEI Regression", train_pear, e)
        writer.add_scalar("Test Pearson - MOSEI Regression", test_pear, e)
        if best_loss == None or best_loss > test_loss:
            best_loss, best_label, best_pred, best_mask, best_pear =\
                    test_loss, test_label, test_pred, test_mask, test_pear
            torch.save(model.state_dict(), args.model_path)

    print('Model saved at {}'.format(args.model_path), file=sys.stderr)

    print('Test performance..')
    print('Loss {} MAE {} r {}'.format(best_loss, round(mean_absolute_error(best_label,best_pred,sample_weight=best_mask),4), best_pear))
