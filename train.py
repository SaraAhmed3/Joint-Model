import joint_model
import dataloader
import train_iter
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def start_train(checkpoint_file = None):
    hidden_size = 1024
    output_size_shape = 6
    output_size_size = 4
    output_size_color = 7
    output_size_action = 5
    output_size_type  = 3
    batch_size = 512

    attn_decoder = joint_model.AttnDecoderRNN(hidden_size, output_size_shape, output_size_size, output_size_color,
                 output_size_action, output_size_type, dropout_p=0.1).to(device)
    
    decoder_optimizer = optim.SGD(attn_decoder.parameters(), lr=0.01)

    if checkpoint_file is not None:
        checkpoint = torch.load(checkpoint_file, map_location=('cpu'))
        attn_decoder.load_state_dict(checkpoint['state_dict'])
        decoder_optimizer.load_state_dict(checkpoint['optimizer'])
        
    attn_decoder.train()
    output_folder = 'D:/uOttawa/final_project/pickle'
    shapes_dataset = dataloader.ShapeDataset(pickle_dir = output_folder)
    shapes_dataloader = DataLoader(shapes_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
           
    # dataloader = getdataloader(type, batch_size,output_folder)
    train_iter.trainIters(shapes_dataloader, attn_decoder, decoder_optimizer,2000) 
