import torch
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import time 


class Trainer:
    def __init__(self, dataset, model, str_optimizer, runs, epochs, lr, weight_decay, early_stopping, logger, momentum, eps, update_freq, gamma, alpha, hyperparam):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.model = model
        self.str_optimizer = str_optimizer
        self.runs = runs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.logger = logger
        self.momentum = momentum
        self.eps = eps
        self.update_freq = update_freq
        self.gamma = gamma
        self.alpha = alpha
        self.hyperparam = hyperparam
        self.path_runs = "runs"

    def train(self):
        if self.logger is not None:
            if self.hyperparam:
                self.logger += f"-{self.hyperparam}{eval(self.hyperparam)}"
            path_logger = os.path.join(self.path_runs, self.logger)
            print(f"path logger: {path_logger}")

            self.empty_dir(path_logger)
            self.logger = SummaryWriter(log_dir=os.path.join(self.path_runs, self.logger)) if self.logger is not None else None

        val_losses, accs, durations = [], [], []
        torch.manual_seed(42)
        for i_run in range(self.runs):
            data = self.dataset[0]
            data = data.to(self.device)

            self.model.to(self.device).reset_parameters()

            optimizer = self.get_optimizer()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_start = time.perf_counter()

            best_val_loss = float('inf')
            test_acc = 0
            val_loss_history = []

            for epoch in range(1, self.epochs + 1):
                lam = (float(epoch)/float(self.epochs))**self.gamma if self.gamma is not None else 0.
                self.train_epoch(self.model, optimizer, data, lam)
                eval_info = self.evaluate_epoch(self.model, data)
                eval_info['epoch'] = int(epoch)
                eval_info['run'] = int(i_run+1)
                eval_info['time'] = time.perf_counter() - t_start
                eval_info['eps'] = self.eps
                eval_info['update-freq'] = self.update_freq

                if self.gamma is not None:
                    eval_info['gamma'] = self.gamma

                if self.alpha is not None:
                    eval_info['alpha'] = self.alpha

                if self.logger is not None:
                    for k, v in eval_info.items():
                        self.logger.add_scalar(k, v, global_step=epoch)

                if eval_info['val loss'] < best_val_loss:
                    best_val_loss = eval_info['val loss']
                    test_acc = eval_info['test acc']

                val_loss_history.append(eval_info['val loss'])
                if self.early_stopping > 0 and epoch > self.epochs // 2:
                    tmp = torch.tensor(val_loss_history[-(self.early_stopping + 1):-1])
                    if eval_info['val loss'] > tmp.mean().item():
                        break
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_end = time.perf_counter()

            val_losses.append(best_val_loss)
            accs.append(test_acc)
            durations.append(t_end - t_start)

        if self.logger is not None:
            self.logger.close()
        loss, acc, duration = torch.tensor(val_losses), torch.tensor(accs), torch.tensor(durations)
        print('Val Loss: {:.4f}, Test Accuracy: {:.2f} ± {:.2f}, Duration: {:.3f} \n'.
              format(loss.mean().item(),
                     100*acc.mean().item(),
                     100*acc.std().item(),
                     duration.mean().item()))

    def train_epoch(self, model, optimizer, data, preconditioner=None, lam=0.):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        label = out.max(1)[1]
        label[data.train_mask] = data.y[data.train_mask]
        label.requires_grad = False

        loss = F.nll_loss(out[data.train_mask], label[data.train_mask])
        loss += lam * F.nll_loss(out[~data.train_mask], label[~data.train_mask])

        loss.backward(retain_graph=True)
        if preconditioner:
            preconditioner.step(lam=lam)
        optimizer.step()

    def evaluate_epoch(self, model, data):
        model.eval()

        with torch.no_grad():
            logits = model(data)

        outs = {}
        for key in ['train', 'val', 'test']:
            mask = data['{}_mask'.format(key)]
            loss = F.nll_loss(logits[mask], data.y[mask]).item()
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            outs['{} loss'.format(key)] = loss
            outs['{} acc'.format(key)] = acc

        return outs

    def get_optimizer(self):
        if self.str_optimizer == 'Adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.str_optimizer == 'SGD':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
            )

    def empty_dir(self, path):
        if os.path.exists(path):
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        else:
            os.makedirs(path)