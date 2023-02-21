import torch
from functorch import jacfwd, vmap

def jac(f, z):
    # composed with vmap for batched Jacobians
    return vmap(jacfwd(f))(z)
    
def jac_robust(f, z):
    # alternative jac if experiencing crashes 
    batch_size, z_dim = z.size()
    v = torch.eye(z_dim).unsqueeze(0).repeat(batch_size, 1, 1).view(-1, z_dim).to(z)
    z = z.repeat(1, z_dim).view(-1, z_dim)
    return torch.autograd.functional.jvp(f, z, v=v)[1].view(batch_size, z_dim, -1).permute(0, 2, 1)

def C_matrix(f, z, normalize=False):
    '''
    f:      model.decode: (b, m) -> (b, n) 
    z:      torch.tensor whose size = (b, m) (keep b low for memory)
    out:    torch.tensor whose size = (b, n, n)
    Outputs the gene "correlation" matrix C by batch J @ J.T
    '''
    J = jac(f, z)
    if normalize:
        return torch.bmm(J, J.transpose(1,2)) / J.norm(p=2, dim=2).unsqueeze(-1)
    else:
        return torch.bmm(J, J.transpose(1,2))


def get_Riemannian_metric(f, z):
    '''
    f:      model.decode: function (b, m) -> (b, n)
    z:      torch.tensor whose size = (b, m)
    out:    torch.tensor whose size = (b, n, n)
    Outputs the Riemannian metric on the latent space 
    '''
    J = jac_robust(f, z)
    out = torch.einsum('nij,nik->njk', J, J)  #J.T @ J instead! 
    return out

# from https://github.com/Gabe-YHLee/GM4HDDA
def compute_length_of_curve(curve, pretrained_model, get_Riemannian_metric):
    '''
    curve:  torch.tensor whose size = (L, d)
    out:    torch.tensor whose size = (1)
    Outputs the length of the curve
    '''
    dt = 1/(len(curve)-1)
    velocity = (curve[:-1] - curve[1:])/dt # (L-1, d)
    G = get_Riemannian_metric(pretrained_model.decode, curve[:-1]) # (L-1, d, d)
    out = torch.sqrt(torch.einsum('ni, nij, nj -> n', velocity, G, velocity)).sum() * dt # int(sqrt(velocity.T @ G @ velocity)) dt
    return out    

# from https://github.com/Gabe-YHLee/GM4HDDA
def compute_geodesic(z1, z2, pretrained_model, get_Riemannian_metric, num_discretization=100, dev=f"cuda:{0}"):
    '''
    z1 : torch.tensor whose size = (1, 2)
    z1 : torch.tensor whose size = (1, 2)
    out: torch.tensor whose size = (num_discretization, 2)
    Outputs the geodesic curve between z1 and z2
    '''
    from scipy.optimize import minimize
    class GeodesicFittingTool():
        def __init__(self, z1, z2, z_init, pretrained_model, get_Riemannian_metric, num_discretization, method, device=dev):
            self.z1 = z1
            self.z2 = z2
            self.pretrained_model = pretrained_model
            self.get_Riemannian_metric = get_Riemannian_metric
            self.num_discretization = num_discretization
            self.delta_t = 1/(num_discretization-1)
            self.device = device
            self.method = method
            self.z_init_input = z_init
            self.initialize()
            
        def initialize(self):
            self.z_init= self.z1.squeeze(0)
            self.z_final= self.z2.squeeze(0)
            dim = self.z_final.size(0)
            self.init_z = self.z_init_input.detach().cpu().numpy()
            self.z_shape = self.init_z.shape
            self.init_z_vec = self.init_z.flatten()

        def geodesic_loss(self, z): 
            z_torch = torch.tensor(z.reshape(self.z_shape), dtype=torch.float32).to(self.device)
            z_extended = torch.cat([self.z_init.unsqueeze(0), z_torch, self.z_final.unsqueeze(0)], dim=0)
            G_ = self.get_Riemannian_metric(self.pretrained_model.decode, z_extended[:-1])
            delta_z = (z_extended[1:, :]-z_extended[:-1, :])/(self.delta_t)
            loss = torch.einsum('ni, nij, nj -> ', delta_z, G_, delta_z) * self.delta_t
            return loss.item()
        
        def jac(self, z):
            z_torch = torch.tensor(z.reshape(self.z_shape), dtype=torch.float32).to(self.device)
            z_torch.requires_grad = True
            z_extended = torch.cat([self.z_init.unsqueeze(0), z_torch, self.z_final.unsqueeze(0)], dim=0)
            G_ = self.get_Riemannian_metric(self.pretrained_model.decode, z_extended[:-1], create_graph=True)
            delta_z = (z_extended[1:, :]-z_extended[:-1, :])/(self.delta_t)
            loss = torch.einsum('ni, nij, nj -> ', delta_z, G_, delta_z) * self.delta_t
            loss.backward()
            z_grad = z_torch.grad
            return z_grad.detach().cpu().numpy().flatten()

        def callback(self, z):
            self.Nfeval += 1
            return print('{} th loss : {}'.format(self.Nfeval, self.geodesic_loss(z)))
            
        def BFGS_optimizer(self, callback=False, maxiter=1000):
            self.Nfeval = 0
            z0 = self.init_z_vec
            if callback == True:
                call = self.callback
            else:
                call = None
            res = minimize(
                self.geodesic_loss, 
                z0, 
                callback=call, 
                method=self.method,
                jac = self.jac,
                options = {
                    'gtol': 1e-10, 
                    'eps': 1.4901161193847656e-08, 
                    'maxiter': maxiter, 
                    'disp': True, 
                    'return_all': False, 
                    'finite_diff_rel_step': None}
                )
            self.res = res

    z12_linear_curve = torch.cat([z1.to(dev) + (z2.to(dev) - z1.to(dev)) * t/(num_discretization-1) for t in range(num_discretization)], dim=0)
    
    tool = GeodesicFittingTool(z1, z2, z12_linear_curve[1:-1], pretrained_model, get_Riemannian_metric, num_discretization, 'BFGS', device=dev)
    tool.BFGS_optimizer()
    z_torch = torch.tensor(tool.res['x'].reshape(tool.z_shape), dtype=torch.float32).to(dev)
    out = torch.cat([tool.z_init.unsqueeze(0), z_torch, tool.z_final.unsqueeze(0)], dim=0)
    return out
