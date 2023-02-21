from datetime import datetime
from torch import optim
from models import VAE, GMVAE
import os
import torch


# save model and include timestamp
def save_model(model, optimizer, loss_hist, path):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
        
    full_path = f"checkpoints/{path}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_hist': loss_hist
    }, full_path)
    print(f"Model saved to {full_path}.")

def load_model(path, model_type, model_params, dev):
    # check if path is a full path or just a filename
    if not os.path.exists(path):
        path = f"checkpoints/{path}"
    checkpoint = torch.load(path, map_location=dev)
    # create model based on model_type
    if model_type == "gmvae":
        model = GMVAE(**model_params).to(dev)
    elif model_type == "vae":
        model = VAE(**model_params).to(dev)
    else:
        raise ValueError("Model type not recognized.")   

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss_hist = checkpoint['loss_hist']
    return model, optimizer, loss_hist