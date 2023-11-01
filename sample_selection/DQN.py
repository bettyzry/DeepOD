import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


class DQN(nn.Module):
    """
    Deep Q Network
    """

    def __init__(self, n_feature, seq_len, hidden_size, n_actions, device='gpu'):
        super(DQN, self).__init__()
        self.n_feature = n_feature
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_actions = n_actions
        self.device = device
        self.latent = nn.Sequential(
          nn.Linear(n_feature, 35),
          nn.ReLU(),
          nn.Linear(35, 30),
          nn.ReLU(),
          nn.Linear(30, 25),
          nn.ReLU(),
          nn.Linear(25, 20),
          nn.ReLU(),
          nn.Linear(20, 15),
          nn.ReLU(),
          nn.Linear(15, 10),
          nn.ReLU(),
          nn.Linear(10, 5)
        )
        # self.output_layer = nn.Linear(self.seq_len*self.hidden_size, n_actions)
        self.output_layer = nn.Linear(5, n_actions)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        x = F.relu(self.latent(x))
        x = self.output_layer(x)
        x = torch.sum(x, dim=1)
        return x

    def get_latent(self, x):
        """
        Get the latent representation of the input using the latent layer
        """
        self.eval()  # 关闭dropout层
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            latent_embs = F.relu(self.latent(x))
        self.train()
        return latent_embs
    #
    # def predict_label(self, x):
    #     self.eval()
    #     """
    #     Predict the label of the input as the argmax of the output layer
    #     """
    #     if not isinstance(x, torch.Tensor):
    #         x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
    #
    #     with torch.no_grad():
    #         ret = torch.argmax(self.forward(x), axis=1)
    #         self.train()
    #         return ret
    #
    # def _initialize_weights(self, ):
    #     with torch.no_grad():
    #         for m in self.modules():
    #             if isinstance(m, nn.Linear):
    #                 nn.init.normal_(m.weight, 0.0, 0.01)
    #                 nn.init.constant_(m.bias, 0.0)
    #
    # def forward_latent(self, x):
    #     if not isinstance(x, torch.Tensor):
    #         x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
    #     latent = F.relu(self.latent(x))
    #     out = self.output_layer(latent)
    #     return out, latent
    #
    # def get_latent_grad(self, x):
    #     if not isinstance(x, torch.Tensor):
    #         x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
    #     latent_embs = F.relu(self.latent(x))
    #     return latent_embs
