import numpy as np, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
import argparse, time, pickle, os, sys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
from model import CategoricalModel, MaskedNLLLoss, BiModalAttention
from dataloader import MOSEICategorical

np.random.seed(393)
torch.cuda.device([0])

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_MOSEI_loaders(path, batch_size=128, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MOSEICategorical(path=path)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, collate_fn=trainset.collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler, collate_fn=trainset.collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    testset = MOSEICategorical(path=path, train=False)
    test_loader = DataLoader(testset, batch_size=batch_size, collate_fn=testset.collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader

def train_or_eval_model(model,loss_function, dataloader, epoch, optimizer=None, train=False, cuda=True):
    count = 0
    losses, preds, labels, masks, alphas_f, alphas_b, vids = [], [], [], [], [], [], []
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    for data in dataloader:
        count+=1
        if train:
            optimizer.zero_grad()
        textf, visuf, acouf, qmask, umask, label =  [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        log_prob ,alpha_f,alpha_b  = model(textf, acouf, visuf, textf, qmask, umask)   
        lp_ = log_prob.transpose(0,1).contiguous().view(-1,log_prob.size()[2]) 
        labels_ = label.view(-1) 
        loss = loss_function(lp_, labels_, umask)
        pred_ = torch.argmax(lp_,1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            optimizer.step()
        else:
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]
    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'),[]
    avg_loss = round(np.sum(losses)/np.sum(masks),4)
    avg_accuracy = round(accuracy_score(labels,preds,sample_weight=masks)*100,2)
    avg_fscore = round(f1_score(labels,preds,sample_weight=masks,average='weighted')*100,2)
    return avg_loss, avg_accuracy, labels, preds, masks,avg_fscore, [alphas_f, alphas_b, vids]

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Trains a categorical model for sentiment data with 1 as positive sentiment and 0 as negative sentiment")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=128, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=50, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=True, help='class weight')
    parser.add_argument('--log_dir', type=str, default='logs/mosei_categorical', help='Directory for tensorboard logs')
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
    n_classes  = 2
    cuda       = args.cuda
    n_epochs   = args.epochs
    D_m_text, D_m_audio, D_m_video, D_m_context = 300, 384, 35, 300
    D_g, D_p, D_e, D_h, D_a = 150, 150, 100, 100, 100

    # Instantiate model
    model = CategoricalModel(D_m_text, D_m_audio, D_m_video, D_m_context, D_g, D_p, D_e, D_h, n_classes=n_classes, dropout_rec=args.rec_dropout, dropout=args.dropout)
    
    if cuda:
        model.cuda()
    loss_weights = torch.FloatTensor([1/0.3097, 1/0.6903])
    if args.class_weight:
        loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = MaskedNLLLoss()
    
    # Get optimizer and  relevant dataloaders
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    train_loader, valid_loader, test_loader = get_MOSEI_loaders('./data/categorical.pkl', valid=0.0, batch_size=batch_size, num_workers=0)
    best_loss, best_label, best_pred, best_mask = None, None, None, None

    # Training loop
    for e in tqdm(range(n_epochs), desc = 'MOSEI Categorical'):
        train_loss, train_acc, _,_,_,train_fscore,_ = train_or_eval_model(model, loss_function, train_loader, e, optimizer, True, cuda=cuda)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model, loss_function, test_loader, e, cuda=cuda)
        writer.add_scalar("Train Loss - MOSEI Categorical", train_loss, e)
        writer.add_scalar("Test Loss - MOSEI Categorical", test_loss, e)
        if best_loss == None or best_loss > test_loss:
            best_loss, best_label, best_pred, best_mask, best_attn = test_loss, test_label, test_pred, test_mask, attentions
            torch.save(model.state_dict(), args.model_path)

    print('Model saved at {}'.format(args.model_path), file=sys.stderr)

    print('Test performance...')
    print('Loss {} accuracy {}'.format(best_loss, round(accuracy_score(best_label,best_pred,sample_weight=best_mask)*100,2)))
    print(classification_report(best_label,best_pred,sample_weight=best_mask,digits=4))
    print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))
