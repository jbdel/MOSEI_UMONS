import torch.optim as Optim


class WarmupOptimizer(object):
    def __init__(self, lr_base, optimizer, data_size, batch_size, warmup_epoch):
        self.optimizer = optimizer
        self._step = 0
        self.lr_base = lr_base
        self._rate = 0
        self.data_size = data_size
        self.batch_size = batch_size
        self.warmup_epoch = warmup_epoch


    def step(self):
        self._step += 1

        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate

        self.optimizer.step()


    def zero_grad(self):
        self.optimizer.zero_grad()


    def rate(self, step=None):
        if step is None:
            step = self._step

        if step <= int(self.data_size / self.batch_size * (self.warmup_epoch + 1) * 0.25):
            r = self.lr_base * 1/(self.warmup_epoch + 1)
        elif step <= int(self.data_size / self.batch_size * (self.warmup_epoch + 1) * 0.5):
            r = self.lr_base * 2/(self.warmup_epoch + 1)
        elif step <= int(self.data_size / self.batch_size * (self.warmup_epoch + 1) * 0.75):
            r = self.lr_base * 3/(self.warmup_epoch + 1)
        else:
            r = self.lr_base

        return r

    def __str__(self):
        return "optimizer -> " + str(self.optimizer)



def get_optim(args, model, data_size, lr_base=None):
    if lr_base is None:
        lr_base = args.lr_base

    std_optim = getattr(Optim, args.opt)
    params = filter(lambda p: p.requires_grad, model.parameters())
    eval_str = 'params, lr=0'
    d_opt = eval(args.opt_params)
    for key in d_opt:
        eval_str += ' ,' + key + '=' + str(d_opt[key])

    optim = WarmupOptimizer(
        lr_base,
        eval('std_optim' + '(' + eval_str + ')'),
        data_size,
        args.batch_size,
        args.warmup_epoch
    )

    return optim


def adjust_lr(optim, decay_r):
    optim.lr_base *= decay_r