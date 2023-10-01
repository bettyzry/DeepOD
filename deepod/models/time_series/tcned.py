"""
TCN is adapted from https://github.com/locuslab/TCN
"""
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from deepod.core.base_model import BaseDeepAD
from deepod.core.networks.ts_network_tcn import TcnAE
from deepod.utils.utility import get_sub_seqs, get_sub_seqs_label
from deepod.metrics import ts_metrics, point_adjustment


class TcnED(BaseDeepAD):
    def __init__(self, seq_len=100, stride=1, epochs=10, batch_size=32, lr=1e-4,
                 rep_dim=32, hidden_dims=32, kernel_size=3, act='LeakyReLU', bias=False, dropout=0.2,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        super(TcnED, self).__init__(
            model_name='TcnED', data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
            seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.act = act
        self.bias = bias

        return

    def fit(self, X, y=None, Xtest=None, Ytest=None, X_seqs=None, y_seqs=None):
        """
        Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : numpy array of shape (n_samples, )
            Not used in unsupervised methods, present for API consistency by convention.
            used in (semi-/weakly-) supervised methods

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if self.data_type == 'ts':
            if X_seqs is not None and y_seqs is not None:
                pass
            else:
                if self.sample_selection == 4 or self.sample_selection == 7:
                    self.ori_data = X
                    self.seq_starts = np.arange(0, X.shape[0] - self.seq_len + 1, self.seq_len)     # 无重叠计算seq
                    self.trainsets['seqstarts0'] = self.seq_starts
                    X_seqs = np.array([X[i:i + self.seq_len] for i in self.seq_starts])
                    y_seqs = get_sub_seqs_label(y, seq_len=self.seq_len, stride=self.seq_len) if y is not None else None
                else:
                    X_seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=self.stride)
                    y_seqs = get_sub_seqs_label(y, seq_len=self.seq_len, stride=self.stride) if y is not None else None
            self.train_data = X_seqs
            self.train_label = y_seqs
            self.n_samples, self.n_features = X_seqs.shape[0], X_seqs.shape[2]
            if self.train_label is not None:
                self.trainsets['yseq0'] = self.train_label
                self.ori_label = y
        else:
            self.train_data = X
            self.train_label = y
            self.n_samples, self.n_features = X.shape

        if self.verbose >= 1:
            print('Start Training...')

        if self.n_ensemble == 'auto':
            self.n_ensemble = int(np.floor(100 / (np.log(self.n_samples) + self.n_features)) + 1)
        if self.verbose >= 1:
            print(f'ensemble size: {self.n_ensemble}')

        for _ in range(self.n_ensemble):
            self.train_loader, self.net, self.criterion = self.training_prepare(self.train_data,
                                                                            y=self.train_label)
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                         lr=self.lr,
                                         eps=1e-6)

            self.net.train()
            for epoch in range(self.epochs):
                t1 = time.time()
                loss = self.training(epoch)
                self.do_sample_selection(epoch)

                print(f'epoch{epoch + 1:3d}, '
                      f'training loss: {loss:.6f}, '
                      f'time: {time.time() - t1:.1f}s')

                if Xtest is not None and Ytest is not None:
                    self.net.eval()
                    scores = self.decision_function(Xtest)
                    eval_metrics = ts_metrics(Ytest, scores)
                    adj_eval_metrics = ts_metrics(Ytest, point_adjustment(Ytest, scores))
                    result = [eval_metrics[0], eval_metrics[1], eval_metrics[2], adj_eval_metrics[0], adj_eval_metrics[1], adj_eval_metrics[2]]
                    print(result)
                    self.result_detail.append(result)
                    self.net.train()
        return

    def training(self, epoch):
        total_loss = 0
        cnt = 0
        self.net.zero_grad()
        for batch_x in self.train_loader:
            batch_x = batch_x.float().to(self.device)
            loss = self.training_forward(batch_x, self.net, self.criterion)
            loss.backward()
            if self.sample_selection == 5:      # ICML21
                to_concat_g = []
                to_concat_v = []
                clip = 0.2
                for name, param in self.net.named_parameters():
                    to_concat_g.append(param.grad.data.view(-1))
                    to_concat_v.append(param.data.view(-1))
                all_g = torch.cat(to_concat_g)
                all_v = torch.cat(to_concat_v)
                metric = torch.abs(all_g * all_v)
                num_params = all_v.size(0)
                nz = int(clip * num_params)
                top_values, _ = torch.topk(metric, nz)
                thresh = top_values[-1]

                for name, param in self.net.named_parameters():
                    mask = (torch.abs(param.data * param.grad.data) >= thresh).type(torch.cuda.FloatTensor)
                    mask = mask * clip
                    param.grad.data = mask * param.grad.data

            self.optimizer.step()
            self.net.zero_grad()
            total_loss += loss.item()
            cnt += 1

            # terminate this epoch when reaching assigned maximum steps per epoch
            if cnt > self.epoch_steps != -1:
                break

        self.epoch_update()
        return total_loss / cnt

    def training_prepare(self, X, y):
        train_loader = DataLoader(X, batch_size=self.batch_size, shuffle=True, drop_last=False)

        net = TcnAE(
            n_features=self.n_features,
            n_hidden=self.hidden_dims,
            n_emb=self.rep_dim,
            activation=self.act,
            bias=self.bias,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        ).to(self.device)

        criterion = torch.nn.MSELoss(reduction="mean")

        if self.verbose >= 2:
            print(net)

        return train_loader, net, criterion

    def inference_prepare(self, X):
        test_loader = DataLoader(X, batch_size=self.batch_size,
                                 drop_last=False, shuffle=False)
        self.criterion.reduction = 'none'
        return test_loader

    def training_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)
        output, _ = net(batch_x)
        # loss = torch.nn.MSELoss(reduction='mean')(output[:, -1], batch_x[:, -1])
        loss = torch.nn.MSELoss(reduction='none')(output[:, -1], batch_x[:, -1])
        loss = loss.mean(axis=1)
        loss = loss.mean()
        return loss

    def inference_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)
        output, _ = net(batch_x)
        error = torch.nn.MSELoss(reduction='none')(output[:, -1], batch_x[:, -1])
        error = error.mean(axis=1)
        return output, error
