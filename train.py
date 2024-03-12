import torch
import torch.nn as nn
from models.vae import VAE
from src.DataLoader import Dataset
import argparse
import os

def arguments():
    parser = argparse.ArgumentParser(description= 'add hypeparameters for training')
    parser.add_argument('--batch_size', type = int, default = 128, help= 'batch-size for training data' )
    parser.add_argument('--input_chanel', type = int, default = 1, help= 'no. of input chanels' )
    parser.add_argument('--epochs', type = int, default = 100, help = 'total number of epochs')
    parser.add_argument('--lr', type = float, default = 0.003, help = 'learning rate')
    parser.add_argument('--latent_dimension', type = int, default = 200, help = 'latent vector dimension')
    parser.add_argument("--device", default="cuda:0", help="cuda device, i.e. 0 or cpu")
    return parser.parse_args()
    
def main():   
    args = arguments()
    model = VAE(args.input_chanel,args.latent_dimension)
    model = model.to(args.device)
    criterion = nn.BCELoss(reduction="sum")
    opt = torch.optim.Adam(model.parameters(),lr = args.lr)
    train_loader , val_loader = Dataset(args.batch_size)
    os.makedirs('./weight', exist_ok=True) 
    best_val_loss = 10000000
    for epoch in range(args.epochs):
        model.train(True)
        total_loss_train = 0.0
        last_loss_train = 0.0
        total_loss_val = 0.0 
        
        for id,(batch_images,batch_labels) in enumerate(train_loader):
            batch_images = batch_images.to(args.device)
            reconstructed_batch_images, mu, log_var,latent_vectors = model(batch_images)
            opt.zero_grad()
            reconstructed_loss = criterion(reconstructed_batch_images,batch_images)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            total_vae_loss = reconstructed_loss + kl_loss
            total_loss_train += total_vae_loss.item()
            total_vae_loss.backward()
            opt.step()
            
            if id % 100 == 99:
                last_loss_train = total_loss_train / 100
                total_loss_train = 0.0
                print('iteration {} loss: {}'.format(id + 1, last_loss_train))
                
        model.eval()
        with torch.no_grad():
            for idx_val , (batch_val_images , batch_val_labels) in enumerate(val_loader):
                batch_val_images = batch_val_images.to(args.device)
                batch_prediction_val, mu_val, log_var_val,latent_vectors_val = model(batch_val_images)
                loss_val = criterion(batch_prediction_val, batch_val_images)
                kl_loss_val = -0.5 * torch.sum(1 + log_var_val - mu_val.pow(2) - log_var_val.exp())
                val_loss = loss_val + kl_loss_val
                total_loss_val += val_loss.item()
            
        val_loss_per_epoch = total_loss_val / len(val_loader)    
        if val_loss_per_epoch < best_val_loss:
            best_val_loss = val_loss_per_epoch
            torch.save(model.state_dict() , os.path.join("./weight" , 'vae.pt'))
            
        print(f'Epochs: {epoch + 1} | Train Loss: {last_loss_train: .3f} | Val Loss: {val_loss_per_epoch : .3f}' )
        
    
    
if __name__ == '__main__':
    main()