import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA
import numpy as np

def loss_cls(alpha, m):
    # alpha(N, J)
    # m(N, J)
    rates = alpha / alpha.sum(dim=1, keepdim=True)
    return - torch.sum(m * rates.log())

from torch.distributions.kl import kl_divergence
from torch.distributions.dirichlet import Dirichlet
def loss_kl(alpha, m):
    # alpha(N, J)
    # m(N, J)
    alpha_ = m + (1-m) * alpha
    m1 = Dirichlet(alpha_)
    m2 = Dirichlet(torch.ones_like(alpha_))
    kl = kl_divergence(m1, m2)
    return torch.mean(kl)


class InvariantModule(nn.Module):
    """Implements an invariant module with keras."""
    
    def __init__(self, in_chn, inv_dim=64):
        super().__init__()
        
        self.s1 = nn.Sequential(
            nn.Linear(in_chn, inv_dim),
            nn.ReLU(),
            nn.Linear(inv_dim, inv_dim),
            nn.ReLU(),
            )
        self.s2 = nn.Sequential(
            nn.Linear(inv_dim, inv_dim),
            nn.ReLU(),
            nn.Linear(inv_dim, inv_dim),
            )
                    
    def forward(self, x):
        """Performs the forward pass of a learnable invariant transform.
        
        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size, N, x_dim)
        
        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, out_dim)
        """
        
        x_reduced = torch.mean(self.s1(x), dim=1) # (batch_size, x_dim)
        out = self.s2(x_reduced)
        return out
    

    
class EquivariantModule(nn.Module):
    """Implements an equivariant module with keras."""
    
    def __init__(self, in_chn, equiv_dim=32, inv_dim=64):
        super().__init__()
        
        self.invariant_module = InvariantModule(in_chn, inv_dim)
        self.s3 = nn.Sequential(
            nn.Linear(inv_dim+in_chn, equiv_dim),
            nn.ReLU(),
            nn.Linear(equiv_dim, equiv_dim),
            )   

    def forward(self, x):
        """Performs the forward pass of a learnable equivariant transform.
        
        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size, N, x_dim)
        
        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, N, equiv_dim)
        """
        
        # Store N
        N = int(x.shape[1])
        
        # Output dim is (batch_size, inv_dim) - > (batch_size, N, inv_dim)
        out_inv = self.invariant_module(x) # (batch_size, inv_dim)
        out_inv_rep = torch.stack([out_inv] * N, dim=1) # (batch_size, N, inv_dim)
        # Concatenate each x with the repeated invariant embedding
        out_c = torch.cat([x, out_inv_rep], dim=-1) # (batch_size, N, x_dim + inv_dim)
        # Pass through equivariant func
        out = self.s3(out_c)
        return out


class InvariantNetwork(nn.Module):
    """Implements an invariant network with keras.
    """

    def __init__(self, in_chn=1, equiv_dim=32, inv_dim=64):
        super().__init__()

        self.equiv_seq = nn.Sequential(
            EquivariantModule(in_chn, equiv_dim, inv_dim),
            EquivariantModule(equiv_dim, equiv_dim, inv_dim)
        )
        self.inv = InvariantModule(equiv_dim, inv_dim)
    
    def forward(self, x):
        """ Performs the forward pass of a learnable deep invariant transformation consisting of
        a sequence of equivariant transforms followed by an invariant transform.
        
        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size, n_obs, data_dim)
        
        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, out_dim + 1)
        """
        
        # Extract n_obs and create sqrt(N) vector
        N = int(x.shape[1])
        N_rep = torch.sqrt(N * torch.ones((x.shape[0], 1), device=x.device)) # (N, 1)

        # Pass through series of augmented equivariant transforms
        out_equiv = self.equiv_seq(x) # (batch_size, N, equiv_dim)

        # Pass through final invariant layer and concatenate with N_rep
        out_inv = self.inv(out_equiv) # (batch_size, out_dim)
        out = torch.cat((out_inv, N_rep), dim=-1) # (out_dim + 1)

        return out



class EvidentialNetwork(nn.Module):

    def __init__(self, in_chn, evid_dim=128, n_models=3):
        """Creates an evidential network and couples it with an optional summary network.

        Parameters
        ----------
        meta : list(dict)
            A list of dictionaries, where each dictionary holds parameter-value pairs
            for a single :class:`tf.keras.Dense` layer
        """

        super().__init__()

        # A network to increase representation power (post-pooling)
        self.dense = nn.Sequential(
            nn.Linear(in_chn, evid_dim),
            nn.ReLU(),
            nn.Linear(evid_dim, evid_dim),
            nn.ReLU(),
            nn.Linear(evid_dim, evid_dim),
            nn.ReLU(),
        )

        # The layer to output model evidences
        self.evidence_layer = nn.Sequential(
            nn.Linear(evid_dim, n_models),
            nn.Softplus(),
            )
        self.J = n_models

    def forward(self, sim_data):
        """Computes evidences for model comparison given a batch of data.

        Parameters
        ----------
        sim_data   : tf.Tensor
            The input where `n_obs` is the ``time`` or ``samples`` dimensions over which pooling is
            performed and ``data_dim`` is the intrinsic input dimensionality, shape (batch_size, n_obs, data_dim)

        Returns
        -------
        alpha      : tf.Tensor of shape (batch_size, n_models) -- the model evidences
        """

        # Compute and return evidence
        return self.evidence(sim_data)

    def predict(self, obs_data, to_numpy=True):
        """Returns the mean, variance and uncertainty implied by the estimated Dirichlet density.

        Parameters
        ----------
        obs_data: tf.Tensor
            Observed data
        to_numpy: bool, default: True
            Flag that controls whether the output is a np.array or tf.Tensor

        Returns
        -------
        out: dict
            Dictionary with keys {m_probs, m_var, uncertainty}
        """

        alpha = self.evidence(obs_data)
        alpha0 = torch.sum(alpha, dim=1, keepdim=True)
        mean = alpha / alpha0
        var = alpha * (alpha0 - alpha) / (alpha0 * alpha0 * (alpha0 + 1))
        uncertainty = self.J / alpha0

        if to_numpy:
            mean = mean.numpy()
            var = var.numpy()
            uncertainty = uncertainty.numpy()

        return {'m_probs': mean, 'm_var': var, 'uncertainty': uncertainty}

    def evidence(self, x):
        """Computes the evidence vector (alpha + 1) as derived from the estimated Dirichlet density.

        Parameters
        ----------
        x  : tf.Tensor
            The conditional data set(s), shape (n_datasets, summary_dim)
        """

        # Pass through dense layer
        x = self.dense(x)

        # Compute evidences
        evidence = self.evidence_layer(x)
        alpha = evidence + 1
        return alpha

    def sample(self, obs_data, n_samples, to_numpy=True):
        """Samples posterior model probabilities from the second-order Dirichlet distro.

        Parameters
        ----------
        obs_data  : tf.Tensor
            The summary of the observed (or simulated) data, shape (n_datasets, summary_dim)
        n_samples : int
            Number of samples to obtain from the approximate posterior
        to_numpy  : bool, default: True
            Flag indicating whether to return the samples as a np.array or a tf.Tensor

        Returns
        -------
        pm_samples : tf.Tensor or np.array
            The posterior samples from the Dirichlet distribution, shape (n_samples, n_batch, n_models)
        """

        # Compute evidential values
        alpha = self.evidence(obs_data)
        n_datasets = alpha.shape[0]

        # Sample for each dataset
        pm_samples = np.stack([np.random.dirichlet(alpha[n, :], size=n_samples) for n in range(n_datasets)], axis=1)

        # Convert to tensor, if specified
        if not to_numpy:
             pm_samples = torch.from_numpy(pm_samples).float()
        return pm_samples

if __name__ == "__main__":
    summary_net = InvariantNetwork(1)
    evidential_net = EvidentialNetwork(65)
    x = torch.randn(2,10,1)
    print(evidential_net(summary_net(x)).shape)