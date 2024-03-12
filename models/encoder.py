import torch
import torch.nn as nn

class Encoder(nn.Module):
  def __init__(self,input_chanel,latent_vector_dim):
    super(Encoder,self).__init__()
    self.conv1 = nn.Conv2d(input_chanel,4,3,bias = False)
    self.conv2 = nn.Conv2d(4,8,3,bias = False)
    self.conv3 = nn.Conv2d(8,16,3,bias = False)
    self.conv4 = nn.Conv2d(16,16,3,bias = False)
    self.norm1 = nn.BatchNorm2d(4)
    self.norm2 = nn.BatchNorm2d(8)
    self.norm3 = nn.BatchNorm2d(16)
    self.norm4 = nn.BatchNorm2d(16)
    self.flat = nn.Flatten()
    self.dense1 = nn.Linear(6400,400)
    self.mu_latent = nn.Linear(400,latent_vector_dim)
    self.log_var_latent = nn.Linear(400,latent_vector_dim)
    self.act = nn.LeakyReLU()


  

  def forward(self,x):
    x1 = self.conv1(x)
    x2 = self.norm1(x1)
    x3 = self.act(x2)
    x4 = self.conv2(x3)
    x5 = self.norm2(x4)
    x6 = self.act(x5)
    x7 = self.conv3(x6)
    x8 = self.norm3(x7)
    x9 = self.act(x8)
    x10 = self.conv4(x9)
    x11 = self.norm4(x10)
    x12 = self.act(x11)
    x13 = self.flat(x12)
    x14 = self.dense1(x13)
    x15 = self.act(x14)
    mu = self.mu_latent(x15)
    log_var = self.log_var_latent(x15)
    
    return mu,log_var 
   

  
