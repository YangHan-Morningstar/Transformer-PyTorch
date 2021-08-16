class TransformerOptimizer(object):

    def __init__(self, optimizer, d_model, lr_mul=2, n_warmup_steps=40000):
        super(TransformerOptimizer, self).__init__()
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        self.update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def get_lr_scale(self):
        return (self.d_model ** -0.5) * \
               min(self.n_steps ** (-0.5), self.n_steps * self.n_warmup_steps ** (-1.5))

    def update_learning_rate(self):

        self.n_steps += 1
        lr = self.lr_mul * self.get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
