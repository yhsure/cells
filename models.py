import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import geo

# ---------------------------
# - Variational Autoencoder -
# ---------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2) # mu + log_var
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim))
        
    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=-1)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z, library_size=None):
        x_hat = self.decoder(z)
        if library_size is not None:
            x_hat = F.softmax(x_hat, dim=-1) * library_size.unsqueeze(-1)
        return x_hat
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

    def loss(self, x, x_hat, mu, log_var):
        recon_loss = F.mse_loss(x_hat, x, reduction="sum")
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss, kl_div

    # sample around cell
    def sample(self, x, n=10, scale=1., library_size=None):
        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        eps = scale * torch.randn(n, self.latent_dim)
        z = mu + eps * std
        x_hat = self.decode(z, library_size)
        return x_hat, z
    
    # todo: delta which scales w.r.t. size of latent space
    def make_step(self, z, i, delta=0.05, library_size=None):
        '''
        Make a step in the latent space, following the direction 
        of the ith row of the decoder's Jacobian matrix.
        '''
        J = geo.jac(lambda x : self.decode(x, library_size), z)
        z = z - delta * J[:,i,:] # subtract jacobian row corresponding to chosen gene i
        return z



# ------------------------
# - Gaussian Mixture VAE -
# ------------------------
class GMVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_clusters):
        super(GMVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.K = num_clusters
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim*num_clusters*2)
        )
        
        self.decoder = self.decode = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=-1)
        mu = mu.view(-1, self.K, self.latent_dim)
        log_var = log_var.view(-1, self.K, self.latent_dim)
        return mu, log_var

    def reparameterize(self, mu, log_var, temperature=1.0):
        eps = torch.randn_like(log_var)
        z = mu + torch.exp(log_var / 2) * eps
        # Gumbel-Softmax trick 
        gumbel_softmax_logits = (z.view(-1, self.K, self.latent_dim) / temperature).softmax(dim=-1) # (B, K, latent_dim)
        z = (gumbel_softmax_logits * z.view(-1, self.K, self.latent_dim)).sum(dim=1) # (B, latent_dim)
        return z, gumbel_softmax_logits


    def forward(self, x, temperature=1.0):
        x = x.view(-1, self.input_dim)
        mu, log_var = self.encode(x)
        z, gumbel_softmax_logits = self.reparameterize(mu, log_var, temperature=temperature)
        x_hat = self.decode(z)
        return x_hat, mu, log_var, z, gumbel_softmax_logits

    # sample around single data point
    def sample(self, x, temperature=1, n=10, scale=1.):
        # encode
        mu, logvar = self.encode(x)
        std = scale * torch.exp(0.5 * logvar)
        eps = torch.randn(n, self.K, self.latent_dim)
        z = mu + eps * std 

        # apply the Gumbel-Softmax trick
        z = z.view(-1, self.K, self.latent_dim) 
        gumbel_softmax_logits = (z / temperature).softmax(dim=-1)
        z = (gumbel_softmax_logits * z).sum(dim=1)
        
        # more correct?
        # gumbel_softmax_samples = F.gumbel_softmax(gumbel_softmax_logits, tau=1.0, hard=True)
        # z_samples = (gumbel_softmax_samples * z_samples).view(n, -1, self.latent_dim)

        # decode the samples
        x_hat = self.decode(z)
        return x_hat, z
    
    def make_step(self, z, i, delta=0.05):
        '''
        Make a step in the latent space, following the direction 
        of the ith row of the decoder's Jacobian matrix.
        '''
        J = geo.jac(self.decode, z)
        z = z - delta * J[:,i,:] 
        return z



# ---------------------------
# - Deep Generative Decoder -
# ---------------------------
class DGD(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DGD, self).__init__()
        # Use a decoder network and a prior distribution as components
        self.decoder = Decoder(input_dim, hidden_dim, output_dim)
        self.prior = Prior(input_dim)
        self.input_dim = input_dim


    def forward(self, x):
        # Compute the MAP estimate of representation for a given feature vector using gradient ascent
        z = torch.randn(len(x), self.input_dim) # initialize representation randomly
        z.requires_grad_(True)           # enable gradient computation for representation
        opt_z = optim.Adam([z], lr=0.01) # optimizer for representation
        opt_gmm = optim.Adam(self.decoder.parameters(), lr=0.01) # optimizer for decoder 

        for _ in range(10): # Perform 10 steps of gradient ascent
            opt_z.zero_grad() # Zero out gradients
            opt_gmm.zero_grad() 

            # Compute the log joint probability of feature vector and representation using decoder and prior
            log_p_x_z = -F.mse_loss(x, self.decoder(z), reduction='sum') # Use mean squared error loss for feature reconstruction 
            log_p_z = torch.sum(self.prior.log_prob(z)) # Use prior log probability for representation regularization
            
            log_p_x_z_p_z = log_p_x_z + log_p_z

            (-log_p_x_z_p_z).backward() # Compute gradients with respect to negative log joint probability 
            opt_z.step() # Update representation using gradient ascent
            opt_gmm.step() # Update decoder using gradient ascent

        return z

# Define the neural network mapping representations to feature distributions
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the prior distribution over representations
class Prior(nn.Module):
    def __init__(self, input_dim):
        super(Prior, self).__init__()
        # Use a Gaussian distribution with learnable mean and variance
        self.mean = nn.Parameter(torch.zeros(input_dim))
        self.log_var = nn.Parameter(torch.zeros(input_dim))

    def sample(self):
        # Sample from the Gaussian distribution using reparameterization trick
        eps = torch.randn_like(self.mean)
        return self.mean + eps * torch.exp(0.5 * self.log_var)

    def log_prob(self, x):
        # Compute the log probability of a given representation under the Gaussian distribution
        return -0.5 * (torch.log(torch.tensor(2 * torch.pi)) + self.log_var + ((x - self.mean) ** 2) / torch.exp(self.log_var))
