import torch
import torch.nn as nn


import warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_image(target_shape, target_size, target_color, target_action, target_intent, 
                encoder_hidden, encoder_outputs, decoder, decoder_optimizer, criterion_shape, criterion_size,
                criterion_color, criterion_action, criterion_intent):

    decoder_optimizer.zero_grad()

    Batch_size = encoder_hidden.size()[0]

    loss = 0
    correct = 0
   
    decoder_hidden = torch.tensor(encoder_hidden.view(Batch_size,1,-1), device=device) # encoder_hidden
    cell_state = torch.zeros((1,Batch_size,1024), device=device)
    encoder_outputs = torch.tensor(encoder_outputs.view(Batch_size,16,-1), device=device)

   
    output_shape, output_size, output_color, output_action, output_type, decoder_hidden, cell_state, decoder_attention = decoder(
        decoder_hidden, cell_state, encoder_outputs)

        
    _, topi_shape = output_shape.topk(1)
    _, topi_size = output_size.topk(1)
    _, topi_color = output_color.topk(1)
    _, topi_action = output_action.topk(1)
    _, topi_type = output_type.topk(1)
    

    # correct += (topi_shape.eq(torch.tensor(target_shape, device=device))).sum().item()
    # correct += (topi_size.eq(torch.tensor(target_size, device=device))).sum().item()
    # correct += (topi_color.eq(torch.tensor(target_color, device=device))).sum().item()
    # correct += (topi_action.eq(torch.tensor(target_action, device=device))).sum().item()
    # correct += (topi_type.eq(torch.tensor(target_intent, device=device))).sum().item()
    
    correct += torch.eq(target_shape, topi_shape.squeeze()).sum().item()
    correct += torch.eq(target_size, topi_size.squeeze()).sum().item()
    correct += torch.eq(target_color, topi_color.squeeze()).sum().item()
    correct += torch.eq(target_action, topi_action.squeeze()).sum().item()
    correct += torch.eq(target_intent, topi_type.squeeze()).sum().item()
    
    loss += criterion_shape(output_shape, target_shape.long()) 
    loss += criterion_size(output_size, target_size.long())
    loss += criterion_color(output_color, target_color.long())
    loss += criterion_action(output_action, target_action.long())
    loss += criterion_intent(output_type, target_intent.long())
    
    loss.backward()
    decoder_optimizer.step()
    
    return loss / (5*Batch_size), correct / (5*Batch_size)
    
def trainIters(dataloader, decoder, decoder_optimizer,n_iters, print_every=50, plot_every=100, learning_rate=0.01, save_every =130):
    # print_loss_total = 0 
    # print_acc_total = 0
    
    
    criterion_shape = nn.NLLLoss()
    criterion_size = nn.NLLLoss()
    criterion_color = nn.NLLLoss()
    criterion_action = nn.NLLLoss()
    criterion_intent = nn.NLLLoss()
    
    # all_losses = []
    # all_acc = []

    for epoch in range(0, 20):
        print("--------------------",epoch,"-----------------------")
        for iter, sample_batched in enumerate(dataloader):
        
            shape, size, color, action, intent, hidden_state, last_hidden = sample_batched
           
            loss, acc = train_image(shape, size, color, action, intent, 
                  last_hidden, hidden_state, decoder, decoder_optimizer, criterion_shape, criterion_size,
                  criterion_color, criterion_action, criterion_intent)
            print("Loss: ", loss)
            print("acc: ", acc)
        state = {
            'iter': iter,
            'epoch': epoch,
            'state_dict': decoder.state_dict(),
            'optimizer': decoder_optimizer.state_dict(),
            }
        ck_pt = "ck_pt" + str(epoch + 114)
        torch.save(state, 'E:/encoder_decoderV2/model_checkpoints/' + ck_pt + '.pt')
        # if(epoch == 9):
        #   torch.save(decoder, 'final_model.pt')



