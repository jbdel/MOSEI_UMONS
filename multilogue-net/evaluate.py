import torch
import train_categorical
import train_regression
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from model import CategoricalModel, MaskedNLLLoss, RegressionModel, MaskedMSELoss


def eval_categorical(task):
    model_path = './model/{}.model'.format(task)

    D_m_text, D_m_audio, D_m_video, D_m_context = 300, 384, 35, 300
    D_g, D_p, D_e, D_h, D_a = 150, 150, 100, 100, 100

    cuda = torch.cuda.is_available()

    print('Loading model...')
    model = CategoricalModel(D_m_text, D_m_audio, D_m_video, D_m_context,
            D_g, D_p, D_e, D_h, n_classes=2, dropout_rec=0.1, dropout=0.5)
    if cuda:
        model.cuda()
    model.load_state_dict(torch.load(model_path))

    loss_weights = torch.FloatTensor([1/0.3097, 1/0.6903])
    loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)

    print('Evaluating model...')
    _, _, test_loader = train_categorical.get_MOSEI_loaders('./data/categorical.pkl',
            valid=0.0, batch_size=128, num_workers=0)

    avg_loss, avg_accuracy, labels, preds, masks, _, _ = train_categorical.train_or_eval_model(
            model, loss_function, test_loader, None, cuda)
    print('loss =', avg_loss)
    print('accuracy =', avg_accuracy)


def eval_regression(task):
    model_path = './model/{}.model'.format(task)

    D_m_text, D_m_audio, D_m_video, D_m_context = 300, 384, 35, 300
    D_g, D_p, D_e, D_h, D_a = 150, 150, 100, 100, 100

    cuda = torch.cuda.is_available()

    print('Loading model...')
    model = RegressionModel(D_m_text, D_m_audio, D_m_video, D_m_context, 
            D_g, D_p, D_e, D_h, dropout_rec=0.1, dropout=0.25)
    if cuda:
        model.cuda()
    model.load_state_dict(torch.load(model_path))

    loss_function = MaskedMSELoss()

    print('Evaluating model...')
    _, _, test_loader = train_regression.get_MOSEI_loaders('./data/regression.pkl',
            valid=0.0, batch_size=128, num_workers=0)

    _, mae, _, labels, preds, masks, sample_ids = train_regression.train_or_eval_model(
            model, loss_function, test_loader, None, cuda)

    # gather labels and predictions
    df = pd.DataFrame([ (sample_id, label, pred)
        for label, pred, mask, sample_id in zip(labels, preds, masks, sample_ids) if mask == 1],
        columns=['sample_id', 'label', 'pred'])
    df['diff'] = (df.label - df.pred).abs()
    df['label_class'] = df.label.apply(discretize)
    df['pred_class'] = df.pred.apply(discretize)
    df = df.sort_values(by='diff', ascending=False)

    if_correct = df.label_class == df.pred_class

    print('mae =', mean_absolute_error(df.label, df.pred))
    print('acc =', if_correct.sum() / len(if_correct))
    df.to_csv('./analysis/{}.csv'.format(task), index=False)

def discretize(a):
    if a < -2:
        return -3
    if -2 <= a < -1:
        return -2
    if -1 <= a < 0:
        return -1
    if a == 0:
        return 0
    if 0 < a <= 1:
        return 1
    if 1 < a <= 2:
        return 2
    if a > 2:
        return 3

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python evaluate.py <task>')
        sys.exit(0)

    task = sys.argv[1]
    if task.startswith('categorical'):
        eval_categorical(task)
    elif task.startswith('regression'):
        eval_regression(task)
    else:
        raise RuntimeError('Unknown task "{}"'.format(task))

