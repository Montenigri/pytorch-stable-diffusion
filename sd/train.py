import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from diffusion import Diffusion as UNet  
from encoder import VAE_Encoder as Encoder  
from ddpm import DDPMSampler
from tqdm import tqdm
from pipeline import get_time_embedding


def train_unet(num_epochs, train_dataset, val_dataset, batch_size, learning_rate, device, seed=None, n_inference_steps=50):
    # Carica i dati
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = UNet().to(device)
    encoder = Encoder().to(device)
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    sampler = DDPMSampler(generator)
    sampler.set_inference_timesteps(n_inference_steps)
    criterion = nn.L1Loss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)


            inputs = inputs.permute(0, 3, 1, 2)  # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            targets = targets.permute(0, 3, 1, 2)  # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            latents_shape = (inputs.size(0), 4, inputs.size(2) // 2, inputs.size(3) // 2)
            encoder_noise = torch.randn(latents_shape, device=device)
            latents = encoder(inputs, encoder_noise)

            sampler.set_strength(strength=0.1)  # Esempio di strength
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            
            optimizer.zero_grad()
            timesteps = tqdm(sampler.timesteps)
            for timestep in timesteps:
                time_embedding = get_time_embedding(timestep).to(device)
                model_input = latents
                model_output = model(model_input, time_embedding)
                latents = sampler.step(timestep, latents, model_output)

            loss = criterion(latents, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')
        
        # Validazione
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                

                inputs = inputs.permute(0, 3, 1, 2)
                targets = targets.permute(0, 3, 1, 2)  # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)

                latents_shape = (inputs.size(0), 4, inputs.size(2) // 2, inputs.size(3) // 2)
                encoder_noise = torch.randn(latents_shape, device=device)
                latents = encoder(inputs, encoder_noise)


                sampler.set_strength(strength=0.1)
                latents = sampler.add_noise(latents, sampler.timesteps[0])
                
                timesteps = tqdm(sampler.timesteps)
                for timestep in timesteps:
                    time_embedding = get_time_embedding(timestep).to(device)
                    model_input = latents
                    model_output = model(model_input, time_embedding)

                latents = sampler.step(timestep, latents, model_output)

                loss = criterion(latents, targets)
                val_loss += loss.item()
        
        print(f'Validation Loss: {val_loss/len(val_loader)}')
    
    # Salva il modello addestrato
    torch.save(model.state_dict(), 'unet_model.pth')

train_data_path = 'path/to/train/data'  # Sostituisci con il percorso del tuo dataset di addestramento
val_data_path = 'path/to/val/data'  # Sostituisci con il percorso del tuo dataset di validazione
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = CustomDataset(train_data_path)  # Sostituisci con il percorso del tuo dataset di addestramento
val_dataset = CustomDataset(val_data_path)  # Sostituisci con il percorso del tuo dataset di validazione
train_unet(num_epochs=25, train_dataset=train_dataset, val_dataset=val_dataset, batch_size=16, learning_rate=0.001, device=device)