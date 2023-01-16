import numpy as np
from numba import njit
from scipy.stats import truncnorm


class DDM():

    @staticmethod
    def prior(batch_size):
        """
        Samples from the prior 'batch_size' times.
        ----------
        
        Arguments:
        batch_size : int -- the number of samples to draw from the prior
        ----------
        
        Output:
        theta : np.ndarray of shape (batch_size, theta_dim) -- the samples batch of parameters
        """
        
        # Prior ranges for the simulator 
        # v_c ~ U(0.1, 6.0) -> p
        # a_c ~ U(0.1, 4.0) -> A
        # t0 ~ U(0.1, 3.0) -> ter
        p_samples = np.random.uniform(low=(0.0, 0.0, 0.0), 
                                    high=(2.25, 2.4, 0.95), size=(batch_size, 3))
        return p_samples.astype(np.float32)

    @staticmethod
    # @njit
    def diffusion_trial(v, a, ndt, zr, dt, max_steps):
        """Simulates a trial from the diffusion model."""

        n_steps = 0.
        x = a * zr

        # Simulate a single DM path
        while (x > -a and x < a and n_steps < max_steps):

            # DDM equation
            x += v*dt + np.sqrt(dt) * np.random.normal()

            # Increment step
            n_steps += 1.0

        rt = n_steps * dt
        return rt + ndt if x > 0. else -rt - ndt, x / a

    @staticmethod
    # @njit
    def diffusion_condition(n_trials, params, zr=0.0, dt=0.005, max_steps=1e4):
        """Simulates a diffusion process over an entire condition."""
        v, a, ndt = params
        x = np.empty((n_trials, 2))
        for i in range(n_trials):
            x[i] = DDM.diffusion_trial(v, a, ndt, zr, dt, max_steps)
        return x

    @staticmethod
    def batch_simulator(prior_samples, n_obs, dt=0.005, max_iter=3e3, **kwargs):
        """
        Simulate multiple diffusion_model_datasets.
        """
        
        n_sim = prior_samples.shape[0]
        sim_data = np.empty((n_sim, n_obs, 2), dtype=np.float32)

        # Simulate diffusion data
        for i in range(n_sim):
            sim_data[i] = DDM.diffusion_condition(n_obs, prior_samples[i], dt=dt, max_steps=max_iter)
        
        sim_data[:,:,1] = (sim_data[:,:,1] > 0).astype(np.float32)
        return sim_data.astype(np.float32)


class SSP():
    
    @staticmethod
    def prior(batch_size):
        """
        Samples from the prior 'batch_size' times.
        ----------
        
        Arguments:
        batch_size : int -- the number of samples to draw from the prior
        ----------
        
        Output:
        theta : np.ndarray of shape (batch_size, theta_dim) -- the samples batch of parameters
        """
        
        # Prior ranges for the simulator 
        # A ~ U(0, 2.4)
        # ter ~ U(0, 0.95)
        # p ~ U(0, 10)
        # rd ~ U(0, 10)
        # nl ~ U(0, 1), standard deviation of noise
        # sdA ~ U(1, 5)

        # default distribution A ~ U(0, 2.4), ter ~ U(0, 0.95), p ~ U(0, 2.25), rd ~ U(0,1)
        # p_samples = np.random.uniform(low=(0.0, 0.0, 0, 0.5), 
        #                             high=(2.4, 0.95, 2.25, 1.5), size=(batch_size, 4))
        
        # prior distribution provide by Jordan Deakin. 
        # truncated normal, truncnorn.rvs(a, b, /mu, /sigma2) 
        # notice the scipy atrrubution:
        # (a, b = (myclip_a - loc) / scale, (myclip_b - loc) / scale)
        mu = np.array([.2, .4, .5, 70, 4])
        sigma = np.array([.2, .4, .5, 30, 4])
        a = (0 - mu) / sigma
        p_samples = truncnorm.rvs(a, np.inf, mu, sigma, size=(batch_size, 5))
        return p_samples.astype(np.float32)
        
    @staticmethod
    @njit
    def normal_cdf_approximation(x):
        # Choudhury, Amit. "A simple approximation to the area under standard normal curve." 
        # Mathematics and Statistics 2.3 (2014): 147-149.
        # or
        # Yerukala, Ramu, and Naveen Kumar Boiroju. "Approximations to standard normal distribution function." 
        # International Journal of Scientific & Engineering Research 6.4 (2015): 515-518.

        # notice the range of x
        mark = False
        if x < 0:
            x = -x
            mark = True

        Numerator = np.math.exp(-x**2 / 2)
        denominator = 0.226 + 0.64 * x + 0.33 * np.math.sqrt(x**2 + 3)
        cdf = 1 - (1 / np.math.sqrt(2 * np.math.pi)) * Numerator / denominator
        if mark:
            return 1 - cdf
        return cdf

    @staticmethod
    # @njit
    def SSP_diffusion_trial(A, ter, p, rd, sdA, nl, incongruency, dt, max_steps):
        """Simulates a trial from the diffusion model."""

        n_steps = 0.
        x = 0
        # sdA = 1.2
        sdt = sdA

        if incongruency == 1:
            pTarget, pFlanker = p, -p
            

        # Simulate a single DM path
        while (x > -A and x < A and n_steps < max_steps):

            if incongruency == 1:
                # Shrinking Spotlight
                sdt -= (rd * dt)
                sdt = 0.1 if sdt < 0.1 else sdt
                # cdf(xo, mu, std)
                # aTarget = 2 * norm.cdf(0.5, 0, sdt) - 1
                aTarget = 2 * SSP.normal_cdf_approximation(0.5 / sdt) - 1
                aFlanker = 1 - aTarget
                drift_rate = pTarget * aTarget + pFlanker * aFlanker
            else: 
                drift_rate = p

            # DDM equation
            x += drift_rate*dt + nl * np.sqrt(dt) * np.random.normal()

            # Increment step
            n_steps += 1.0

        rt = n_steps * dt
        return rt + ter if x > 0. else -rt - ter, incongruency

    @staticmethod
    # @njit
    def diffusion_condition(n_trials, params, incongruency, dt=0.005, max_steps=3e3):
        """Simulates a diffusion process over an entire condition."""
        A, ter, p, rd, sdA = params
        nl = 1
        # sdA = 1.2
        x = np.empty((n_trials, 2), dtype=np.float32)
        for i in range(n_trials):
            x[i] = SSP.SSP_diffusion_trial(A, ter, p, rd, sdA, nl, incongruency[i], dt, max_steps)
        return x

    @staticmethod
    def batch_simulator(prior_samples, n_obs, incongruency='random', dt=0.005, max_iter=3e3, **kwargs):
        """
        Simulate multiple diffusion_model_datasets.
        input:
        :in: 'random', 0 (inin), 1 (in)
        """
        
        n_sim = prior_samples.shape[0]
        sim_data = np.empty((n_sim, n_obs, 2), dtype=np.float32)

        # in 
        if incongruency == False:
            incongruency = np.zeros((n_sim, n_obs), dtype=np.float32)
        elif incongruency == True:
            incongruency = np.ones((n_sim, n_obs), dtype=np.float32)
        else:
            incongruency = np.random.choice([0, 1], (n_sim, n_obs)).astype(np.float32)

        # Simulate diffusion data
        for i in range(n_sim):
            sim_data[i] = SSP.diffusion_condition(n_obs, prior_samples[i], incongruency[i], dt=dt, max_steps=max_iter)
        
        return sim_data.astype(np.float32)


class DSTP():

    @staticmethod
    def prior(batch_size):
        """
        Samples from the prior 'batch_size' times.
        ----------
        
        Arguments:
        batch_size : int -- the number of samples to draw from the prior
        ----------
        
        Output:
        theta : np.ndarray of shape (batch_size, theta_dim) -- the samples batch of parameters
        """
        
        # Prior ranges for the simulator 
        # A ~ U(0, 2.4)
        # C ~ U(0, 2,4)
        # ter ~ U(0, 0.95)
        # \mu_{ta} ~ U(0, 1.125)
        # \mu_{fl} ~ U(0, 1.125)
        # \mu_{ss} ~ U(0, 2.25)
        # \mu_{rs2} ~ U(\mu_{ta} + \mu_{fl}, 2.25)
        # nl ~ Constant(1), standard deviation of noise

        # p_samples = np.random.uniform(low=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 
        #                             high=(2.4, 2.4, 0.95, 1.125, 1.125, 2.25), size=(batch_size, 6))
        # # low = p_samples[:,3:5].sum(axis=-1, keepdims=True)
        # low = 0
        # high = 2.25
        # mu_rs2 = np.random.random_sample(size=(batch_size, 1)) * (high - low) + low
        # p_samples = np.concatenate([p_samples, mu_rs2], axis=-1)

        mu = np.array([.2, .2, .4, .2, .4, 1.2, .4])
        sigma = np.array([.2, .2, .3, .2, .4, 1.2, .4])
        a = (0 - mu) / sigma
        p_samples = truncnorm.rvs(a, np.inf, mu, sigma, size=(batch_size, 7))
        return p_samples.astype(np.float32)
        
    @staticmethod
    # @njit
    def DSTP_diffusion_trial(A, C, ter, mu_ta, mu_fl, mu_ss, mu_rs2, nl, incongruency, dt, max_steps):
        """Simulates a trial from the diffusion model."""

        n_steps = 0.
        rs = 0 # response selection
        ss = 0 # stimulus selection
        stage = 1 # 1 or 2

        if incongruency == 1:
            mu = mu_ta - mu_fl
        else:
            mu = mu_ta + mu_fl

        # Simulate a single DM path
        while (rs > -A and rs < A and n_steps < max_steps):

            if stage == 1:
            
                # DDM equation
                rs += mu*dt + nl * np.sqrt(dt) * np.random.normal() # Phase 1
                ss += mu_ss*dt + nl * np.sqrt(dt) * np.random.normal() # Phase 2

                if ss < -C or ss > C:
                    stage = 2
                    mu = mu_rs2 if ss > C else -mu_rs2

            if stage == 2:
                # DDM equation
                rs += mu*dt + nl * np.sqrt(dt) * np.random.normal() # Phase 1

            # Increment step
            n_steps += 1.0

        rt = n_steps * dt
        return rt + ter if rs > 0. else -rt - ter, incongruency

    @staticmethod
    # @njit
    def diffusion_condition(n_trials, params, incongruency, dt=0.005, max_steps=3e3):
        """Simulates a diffusion process over an entire condition."""
        A, C, ter, mu_ta, mu_fl, mu_ss, mu_rs2 = params
        nl = 1
        x = np.empty((n_trials, 2), dtype=np.float32)
        for i in range(n_trials):
            x[i] = DSTP.DSTP_diffusion_trial(
                A, C, ter, mu_ta, mu_fl, mu_ss, mu_rs2, nl, incongruency[i], dt, max_steps)
        return x

    @staticmethod
    def batch_simulator(prior_samples, n_obs, incongruency='random', dt=0.005, max_iter=3e3, **kwargs):
        """
        Simulate multiple diffusion_model_datasets.
        input:
        :in: 'random', 0 (inin), 1 (in)
        """
        
        n_sim = prior_samples.shape[0]
        sim_data = np.empty((n_sim, n_obs, 2), dtype=np.float32)

        # in 
        if incongruency == False:
            incongruency = np.zeros((n_sim, n_obs), dtype=np.float32)
        elif incongruency == True:
            incongruency = np.ones((n_sim, n_obs), dtype=np.float32)
        else:
            incongruency = np.random.choice([0, 1], (n_sim, n_obs)).astype(np.float32)

        # Simulate diffusion data
        for i in range(n_sim):
            sim_data[i] = DSTP.diffusion_condition(n_obs, prior_samples[i], incongruency[i], dt=dt, max_steps=max_iter)
        
        return sim_data.astype(np.float32) # (n_prior, n_obs, n_feature)


def param_padding(params, max_param_length=3):
    n_prior, param_length = params.shape
    assert param_length <= max_param_length
    if param_length == max_param_length:
        return params
    else:
        new_params = np.zeros((n_prior, max_param_length))
        new_params[:,:param_length] = params
        return new_params

def Simulator(
    n_prior=1, n_obs='random', 
    models=[
        {'name': SSP, 'kwargs': {}}, 
        {'name': DSTP, 'kwargs': {}}
        ]
    ):
    """
    priors: list of prior simulator.
    n_obs: number of observations x_i.
    models: list of model prior distribution.

    return: 
        i: model index
        params: (n_prior, n_obs)
        x: (n_prior, n_obs, n_feature)
    """
    n_model = len(models)
    
    if n_obs == 'random':
        n_obs = np.random.randint(5,100)
    if callable(n_obs):
        n_obs = n_obs()

    i = np.random.randint(n_model)
    model = models[i]

    params = model['name'].prior(n_prior)
    x = model['name'].batch_simulator(params, n_obs, **model['kwargs'])

    return i, param_padding(params, max_param_length=7), x


import torch
from torch.distributions.laplace import Laplace
from torch.distributions.normal import Normal
from torch.distributions.studentT import StudentT
from torch.utils.data import DataLoader, Dataset


class SSP_DDM_Dataset(Dataset):

    def __init__(
        self, n_obs='random', length=10000, 
        models=[
        {'name': DDM, 'kwargs': {}}, 
        {'name': SSP, 'kwargs': {}}
        ]
        ):
        super().__init__()
        self.n_obs = n_obs
        self.length = length
        self.models = models

    def __getitem__(self, index):
        i, params, x = Simulator(n_obs=self.n_obs, models=self.models) # scalar, (1, n_obs), (1, n_obs, n_feature)
        # x[:,:,1] = (x[:,:,0] > 0).astype(np.float32) # (pos and neg means acc)
        # x = torch.from_numpy(x[0,:,0]).unsqueeze(dim=-1) # (n_obs, 1)
        x = torch.from_numpy(x[0]) # (n_obs, 2)
        return torch.eye(len(self.models))[i], \
            torch.from_numpy(params[0]), \
            x # (n_obs, *)

    def __len__(self):
        return self.length


class DistributionDataset(Dataset):
    # 1: Normal, loc ~ U(-10,10), scale(0,10)
    # 2: StudentT, loc(-10,10), scale(0,10), df(0, 10)
    # 3: Laplace, loc(-10,10), scale(0,10)
    def __init__(self, n_obs=100):
        super().__init__()
        self.models = [self.sample_Normal, self.sample_Laplace, self.sample_StudentT]
        self.n_obs = n_obs

    @staticmethod
    def sample_model():
        return np.random.choice([0,1,2])
        # return torch.eye(3)[np.random.choice([0,1,2])]   

    @staticmethod
    def sample_StudentT(n_prior=1, n_obs=100):
        # prior
        loc = 20 * torch.rand(n_prior) - 10
        scale = 10 * torch.rand(n_prior)
        df = 10 * torch.rand(n_prior)
        params = torch.stack((loc, scale, df), dim=1) # (n_prior, 3)

        m = StudentT(df, loc, scale) # (n_prior, n_obs)
        return params, m.rsample([n_obs]).T  

    @staticmethod
    def sample_Laplace(n_prior=1, n_obs=100):
        # prior
        loc = 20 * torch.rand(n_prior) - 10
        scale = 10 * torch.rand(n_prior)
        params = torch.stack((loc, scale), dim=1) # (n_prior, 2)

        m = Laplace(loc, scale) # (n_prior, n_obs)
        return params, m.rsample([n_obs]).T  

    @staticmethod
    def sample_Normal(n_prior=1, n_obs=100):
        # prior
        loc = 20 * torch.rand(n_prior) - 10
        scale = 10 * torch.rand(n_prior)
        params = torch.stack((loc, scale), dim=1) # (n_prior, 2)

        m = Normal(loc, scale) # (n_prior, n_obs)
        return params, m.rsample([n_obs]).T  

    def _param_padding(self, params, max_param_length=3):
        n_prior, param_length = params.shape
        assert param_length <= max_param_length
        if param_length == max_param_length:
            return params
        else:
            new_params = torch.zeros((n_prior, max_param_length), device=params.device)
            new_params[:,:param_length] = params
            return new_params

    def __getitem__(self, index):
        if self.n_obs == 'random':
            n_obs = np.random.randint(5,1000)
        index_model = self.sample_model()
        model = self.models[index_model]
        params, data = model(n_obs=n_obs)
        return torch.eye(3)[index_model], \
            self._param_padding(params)[0], \
            data[0].unsqueeze(dim=-1) # (n_obs, 1)   

    def __len__(self):
        return 10000


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # dataset = DistributionDataset()
    # m, p, x = dataset[0]    
    # dataloader = DataLoader(dataset, batch_size=10)
    # for m,p,x in dataloader:
    #     print(m.shape, p.shape, x.shape)
    #     break

    # =================================================
    # params1, data1 = DistributionDataset.sample_Laplace(n_obs=1000)
    # params2, data2 = DistributionDataset.sample_Normal(n_obs=1000)
    # params3, data3 = DistributionDataset.sample_StudentT(n_obs=1000)
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(1,3,figsize=(12,5))
    # axes[0].hist(data1[0], bins=30)
    # axes[0].set_title(f'mu = {params1[0,0]: 0.4f}, \nscale = {params1[0,1]: 0.4f}')

    # axes[1].hist(data2[0], bins=30)
    # axes[1].set_title(f'mu = {params2[0,0]: 0.4f}, \nscale = {params2[0,1]: 0.4f}')
    
    # axes[2].hist(data3[0], bins=30)
    # axes[2].set_title(f'mu = {params3[0,0]: 0.4f}, \nscale = {params3[0,1]: 0.4f}, \n df = {params3[0,2]: 0.4f}')
    # plt.show()


    # =================================================
    models=[
        {'name': SSP, 'kwargs': {'incongruency': 'random'}}, 
    ]
    i, params, x = Simulator(n_prior=100, n_obs=20, models=models)
    x_cong = x[x[:,:,1]==0,0]
    x_incong = x[x[:,:,1]==1,0]
    x_cong_cor = x_cong[x_cong>0]
    x_cong_incor = x_cong[x_cong<0]
    x_incong_cor = x_incong[x_incong>0]
    x_incong_incor = x_incong[x_incong<0]
    print(
        len(x_cong_cor), 
        len(x_cong_incor), 
        len(x_incong_cor), 
        len(x_incong_incor)
    )
    print(
        x_cong_cor.mean(), 
        x_cong_incor.mean(), 
        x_incong_cor.mean(), 
        x_incong_incor.mean()
    )


