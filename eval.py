import torch
from src.DataLoader import Dataset
import os
import cv2
from models.vae import VAE
import numpy as np

_,test_loader = Dataset(128)
total_digits = next(iter(test_loader))[1]
total_images = next(iter(test_loader))[0]

os.makedirs('./result' , exist_ok= True)
loaded_model = VAE(1,200)
loaded_model.load_state_dict(torch.load('./weight/vae.pt'))

loaded_model.eval()
with torch.no_grad():
  for digit in range(10):
    
    list_images = [total_images[j].unsqueeze(0) for j in range(len(total_digits)) if digit==total_digits[j]]
    random_generate_num = np.random.randint(0 , len(list_images))
    
    generated_img , _, _, _ = loaded_model(list_images[random_generate_num])
    
    cv2.imwrite(os.path.join('./result','generated_{}.jpg'.format(digit)),generated_img.permute(2,3,1,0).squeeze(0,-1).cpu().numpy()*255)
    cv2.imshow('',generated_img.permute(2,3,1,0).squeeze(0,-1).cpu().numpy())
    cv2.waitKey(0)

