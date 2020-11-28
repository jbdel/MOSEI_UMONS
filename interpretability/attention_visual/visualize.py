import torch
from utils.compute_args import compute_args
ckpt = 'ckpt/LAV_sentiment_2/best80.75.pkl'
args = torch.load(ckpt)['args']
args = compute_args(args)
evaluation_sets = ['train']
from mosei_dataset import Mosei_Dataset
from model_LAV import Model_LAV
from torch.utils.data import DataLoader
from train import evaluate
from matplotlib import pyplot as plt
# Creating dataloader
valid_dset = eval(args.dataloader)('valid', args)
loaders = {set: DataLoader(eval(args.dataloader)(set, args, valid_dset.token_to_ix),
               args.batch_size,
               num_workers=8,
               pin_memory=True) for set in evaluation_sets}
# Creating net
net = eval(args.model)(args, valid_dset.vocab_size, valid_dset.pretrained_emb).cuda()
state_dict = torch.load(ckpt)['state_dict']
net.load_state_dict(state_dict)
net.train(False)

def draw(att_map, i, name, modality):
  plt.figure()
  plt.subplot(221)
  plt.imshow(att_map[i, 0, :, :], cmap='gray')
  plt.subplot(222)
  plt.imshow(att_map[i, 1, :, :], cmap='gray')
  plt.subplot(223)
  plt.imshow(att_map[i, 2, :, :], cmap='gray')
  plt.subplot(224)
  plt.imshow(att_map[i, 3, :, :], cmap='gray')
  plt.savefig("attention_visual/" + name + "_" + modality + ".jpg")


wrong_ids = ['qgC8_emxSIU[1]', 'fWOIAxzBQFY[6]', 'mmg_eTDHjkk[9]', 'cml9rShionM[3]','XzVapdEr_GY[0]', 'SYQ_zv8dWng[4]']
for step, (ids, x, y, z, ans) in enumerate(loaders['train']):
    x = x.cuda()
    y = y.cuda()
    z = z.cuda()
    pred, att_map_y, att_map_z = net(x, y, z)

    pred = pred.cpu().data.numpy()
    att_map_y = att_map_y.detach().cpu().numpy()
    att_map_z = att_map_z.detach().cpu().numpy()
    for i in range(32):
      print(ids[i])
      name = ids[i]
      if name in wrong_ids:
        draw(att_map_y, i, name, "acoustic")
        draw(att_map_z, i, name, "video")
    ans = ans.cpu().data.numpy()