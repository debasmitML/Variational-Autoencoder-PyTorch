import torch.nn as nn
import torch

class Decoder(nn.Module):
  def __init__(self,latent_vector_dim):
    super(Decoder,self).__init__()
    self.dense_decoder = nn.Linear(latent_vector_dim,6400)
    self.deconv1 = nn.ConvTranspose2d(16,16,3,bias=False)
    self.deconv2 = nn.ConvTranspose2d(16,8,3,bias=False)
    self.deconv3 = nn.ConvTranspose2d(8,4,3,bias=False)
    self.deconv4 = nn.ConvTranspose2d(4,1,3,bias=False)
    self.norm4 = nn.BatchNorm2d(4)
    self.norm3 = nn.BatchNorm2d(8)
    self.norm2 = nn.BatchNorm2d(16)
    self.norm1 = nn.BatchNorm2d(16)
    self.act = nn.LeakyReLU()

  def forward(self,x):
    x1 = self.dense_decoder(x)
    x2 = x1.reshape(-1,16,20,20)
    x3 = self.norm1(x2)
    x4 = self.act(x3)
    x5 = self.deconv1(x4)
    x6 = self.norm2(x5)
    x7 = self.act(x6)
    x8 = self.deconv2(x7)
    x9 = self.norm3(x8)
    x10 = self.act(x9)
    x11 = self.deconv3(x10)
    x12 = self.norm4(x11)
    x13 = self.act(x12)
    out = self.deconv4(x13)
    out1 = torch.sigmoid(out)

    return out1