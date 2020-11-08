import torch
import train_categorical
import train_regression
import sys
from model import CategoricalModel, MaskedNLLLoss, RegressionModel, MaskedMSELoss


def eval_categorical(model_path):
    D_m_text, D_m_audio, D_m_video, D_m_context = 300, 384, 35, 300
    D_g, D_p, D_e, D_h, D_a = 150, 150, 100, 100, 100

    cuda = torch.cuda.is_available()

    print('Loading model...')
    model = CategoricalModel(D_m_text, D_m_audio, D_m_video, D_m_context,
            D_g, D_p, D_e, D_h, n_classes=2, dropout_rec=0.1, dropout=0.5)
    if cuda:
        model.cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    loss_weights = torch.FloatTensor([1/0.3097, 1/0.6903])
    loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)

    print('Evaluating model...')
    _, _, test_loader = train_categorical.get_MOSEI_loaders('./data/categorical.pkl',
            valid=0.0, batch_size=128, num_workers=0)

    avg_loss, avg_accuracy, labels, preds, masks, _, _ = train_categorical.train_or_eval_model(
            model, loss_function, test_loader, None, cuda)
    print('avg loss =', avg_loss)
    print('avg accuracy =', avg_accuracy)


def eval_regression(model_file):
    D_m_text, D_m_audio, D_m_video, D_m_context = 300, 384, 35, 300
    D_g, D_p, D_e, D_h, D_a = 150, 150, 100, 100, 100

    cuda = torch.cuda.is_available()

    print('Loading model...')
    model = RegressionModel(D_m_text, D_m_audio, D_m_video, D_m_context, 
            D_g, D_p, D_e, D_h, dropout_rec=0.1, dropout=0.25)
    if cuda:
        model.cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    loss_function = MaskedMSELoss()

    print('Evaluating model...')
    _, _, test_loader = train_regression.get_MOSEI_loaders('./data/regression.pkl',
            valid=0.0, batch_size=128, num_workers=0)

    avg_loss, mae, pearson, labels, preds, masks = train_regression.train_or_eval_model(
            model, loss_function, test_loader, None, cuda)
    print('avg loss =', avg_loss)
    print('mae =', mae)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python evaluate.py <task>')
        sys.exit(0)

    task = sys.argv[1]
    if task == 'categorical':
        eval_categorical('./model/categorical.model')
    elif task == 'regression':
        eval_regression('./model/regression.model')
    else:
        raise RuntimeError('Unknown task "{}"'.format(task))

