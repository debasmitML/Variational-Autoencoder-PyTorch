from models.encoder import Encoder
from models.decoder import Decoder
import torch.nn as nn
import torch

class VAE(nn.Module):
  def __init__(self,in_channels,latent_vectors_dims):
    super(VAE,self).__init__()
    self.encode = Encoder(in_channels,latent_vectors_dims)
    self.decode = Decoder(latent_vectors_dims)
  
  @staticmethod
  def sampling(latent_mu,latent_log_var):
    
    epsilon = torch.randn_like(latent_mu)
    latent_vectors = latent_mu + torch.exp(latent_log_var/2) * epsilon
    
    return latent_vectors
  
  def forward(self,x):
    
    mu_latent,log_var_latent = self.encode(x)
    latent_vectors = self.sampling(mu_latent,log_var_latent)
    reconstruct = self.decode(latent_vectors)
    
    return reconstruct , mu_latent , log_var_latent, latent_vectors