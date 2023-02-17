# class GaussianMixtureVAE(nn.Module):
#     def __init__(self, input_size, hidden_size, latent_size, num_gaussians):
#         super().__init__()
        
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3_mu = nn.Linear(hidden_size, latent_size)
#         self.fc3_logvar = nn.Linear(hidden_size, latent_size)
#         self.fc4 = nn.Linear(latent_size, hidden_size)
#         self.fc5 = nn.Linear(hidden_size, hidden_size)
#         self.fc6 = nn.Linear(hidden_size, input_size)

#         self.latent_size = latent_size
#         self.num_gaussians = num_gaussians
        
#     def encode(self, x):
#         h = F.relu(self.fc1(x))
#         h = F.relu(self.fc2(h))
#         mu = self.fc3_mu(h)
#         logvar = self.fc3_logvar(h)
#         return mu, logvar

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps*std

#     def decode(self, z):
#         h = F.relu(self.fc4(z))
#         h = F.relu(self.fc5(h))
#         return self.fc6(h)
    
#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar

# def loss_function(x, recon_x, mu, logvar, num_gaussians, beta=1.0):
#     MSE = F.mse_loss(recon_x, x, reduction='sum')
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return MSE + beta * KLD, MSE, KLD

# def train(model, optimizer, train_loader, num_epochs, device, num_gaussians, beta=1.0):
#     model = model.to(device)
#     for epoch in range(num_epochs):
#         train_loss = 0
#         for x, _ in train_loader:
#             x = x.to(device)
#             optimizer.zero_grad()
#             recon_x, mu, logvar = model(x)
#             loss, mse, kld = loss_function(x, recon_x, mu, logvar, num_gaussians, beta)
#             loss.backward()
#             train_loss += loss.item()
#             optimizer.step()
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader.dataset):.4f}, MSE: {mse/len(train_loader.dataset):.4f}, KLD: {kld/len(train_loader.dataset):.4f}')

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GaussianMixtureVAE(input_size=784, hidden_size=512, latent_size=2, num_gaussians=2)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# train(model, optimizer, train_loader, 50, device, 10)


##---------------------------------------------------------------------

# import math

# def loss_fn(recon_X, X, mu_w, logvar_w, qz,
# 	mu_x, logvar_x, mu_px, logvar_px, x_sample, x_size, K):
# 	N = X.size(0) # batch size

# 	# 1. Reconstruction Cost = -E[log(P(y|x))]
# 	# unpack Y into mu_Y and logvar_Y
# 	# mu_recon_X, logvar_recon_X = recon_X

# 	# use gaussian criteria
# 	# negative LL, so sign is flipped
# 	# log(sigma) + 0.5*2*pi + 0.5*(x-mu)^2/sigma^2
# 	# recon_loss = 0.5 * torch.sum(logvar_recon_X + math.log(2*math.pi) \
# 	# 		+ (X - mu_recon_X).pow(2)/logvar_recon_X.exp())
# 	recon_loss = F.mse_loss(recon_X, X)

# 	# 2. KL( q(w) || p(w) )
# 	KLD_W = -0.5 * torch.sum(1 + logvar_w - mu_w.pow(2) - logvar_w.exp())

# 	# 3. KL( q(z) || p(z) )
# 	KLD_Z = torch.sum(qz * torch.log(K * qz + 1e-10))

# 	# 4. E_z_w[KL(q(x)|| p(x|z,w))]
# 	# KL  = 1/2(  logvar2 - logvar1 + (var1 + (m1-m2)^2)/var2  - 1 )
# 	mu_x = mu_x.unsqueeze(-1)
# 	mu_x = mu_x.expand(-1, x_size, K)

# 	logvar_x = logvar_x.unsqueeze(-1)
# 	logvar_x = logvar_x.expand(-1, x_size, K)

# 	# shape (-1, x_size, K)
# 	KLD_QX_PX = 0.5 * (((logvar_px - logvar_x) + \
# 		((logvar_x.exp() + (mu_x - mu_px).pow(2))/logvar_px.exp())) \
# 		- 1)

# 	# transpose to change dim to (-1, x_size, K)
# 	# KLD_QX_PX = KLD_QX_PX.transpose(1,2)
# 	qz = qz.unsqueeze(-1)
# 	qz = qz.expand(-1, K, 1)

# 	E_KLD_QX_PX = torch.sum(torch.bmm(KLD_QX_PX, qz))

# 	# 5. Entropy criterion
	
# 	# CV = H(Z|X, W) = E_q(x,w) [ E_p(z|x,w)[ - log P(z|x,w)] ]
# 	# compute likelihood
	
# 	x_sample = x_sample.unsqueeze(-1)
# 	x_sample =  x_sample.expand(-1, x_size, K)

# 	temp = 0.5 * x_size * math.log(2 * math.pi)
# 	# log likelihood
# 	llh = -0.5 * torch.sum(((x_sample - mu_px).pow(2))/logvar_px.exp(), dim=1) \
# 			- 0.5 * torch.sum(logvar_px, dim=1) - temp

# 	lh = F.softmax(llh, dim=1)

# 	# entropy
# 	CV = torch.sum(torch.mul(torch.log(lh+1e-10), lh))
	
# 	loss = recon_loss + KLD_W + KLD_Z + E_KLD_QX_PX
# 	return loss, recon_loss, KLD_W, KLD_Z, E_KLD_QX_PX, CV


