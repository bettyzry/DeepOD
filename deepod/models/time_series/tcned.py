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

    def fit(self, X, y=None):
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
            X_seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=self.stride)
            y_seqs = get_sub_seqs_label(y, seq_len=self.seq_len, stride=self.stride) if y is not None else None
            self.train_data = X_seqs
            self.train_label = y_seqs
            self.n_samples, self.n_features = X_seqs.shape[0], X_seqs.shape[2]
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
            train_loader, self.net, self.criterion = self.training_prepare(self.train_data,
                                                                            y=self.train_label)
            optimizer = torch.optim.Adam(self.net.parameters(),
                                         lr=self.lr,
                                         weight_decay=1e-5)
            self.net.train()
            for epoch in range(self.epochs):
                self.training(optimizer, train_loader, epoch)

        if self.verbose >= 1:
            print('Start Inference on the training data...')

        self.decision_scores_ = self.decision_function(X)
        self.labels_ = self._process_decision_scores()

        return self

    def training(self, optimizer, dataloader, epoch):
        t1 = time.time()
        total_loss = 0
        cnt = 0
        for batch_x in dataloader:
            loss = self.training_forward(batch_x, self.net, self.criterion)
            self.net.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            cnt += 1

            # terminate this epoch when reaching assigned maximum steps per epoch
            if cnt > self.epoch_steps != -1:
                break

        t = time.time() - t1
        print(f'epoch{epoch + 1:3d}, '
              f'training loss: {total_loss / cnt:.6f}, '
              f'time: {t:.1f}s')

        train_error_now = []
        for batch_x in dataloader:
            _, error = self.inference_forward(batch_x, self.net, self.criterion)
            train_error_now = np.concatenate([train_error_now, error.cpu().detach().numpy()])
        self.loss_by_epoch.append(train_error_now)

        if epoch == 0:
            self.epoch_time = t

        self.epoch_update()

        return total_loss

    def training_prepare(self, X, y):
        train_loader = DataLoader(X, batch_size=self.batch_size, shuffle=True)

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
        ts_batch = batch_x.float().to(self.device)
        output, _ = net(ts_batch)
        loss = criterion(output[:, -1], ts_batch[:, -1])
        return loss

    def inference_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)
        output, _ = net(batch_x)
        error = torch.nn.L1Loss(reduction='none')(output[:, -1], batch_x[:, -1])
        error = torch.sum(error, dim=1)
        return output, error
