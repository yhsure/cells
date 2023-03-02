import torch
import torch.nn as nn
import torch.nn.functional as F
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



