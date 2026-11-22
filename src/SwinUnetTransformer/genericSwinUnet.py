import torch
from torch import nn
import icecream as ic
from src.SwinUnetTransformer.layers import SwinDecoder, SwinEncoder, SwinClassifier

class GenericSwinUnet(nn.Module):
    def __init__(self, params) -> None:
        #self.opt_imgs = len(params['train_opt_imgs'][0])
        #self.sar_imgs = len(params['train_sar_imgs'][0])
        super(GenericSwinUnet, self).__init__()
        self.n_classes = params['n_classes']

        self.base_dim = params['base_dim']
        self.window_size = params['window_size']
        self.shift_size = params['shift_size']
        self.img_size = params['img_size']
        self.patch_size = params['patch_size']
        self.n_heads = params['n_heads']
        self.n_blocks = params['n_blocks']

    def prepare_model(self, in_channels):
        self.encoder = SwinEncoder(
            input_depth = in_channels, 
            base_dim = self.base_dim, 
            window_size = self.window_size,
            shift_size = self.shift_size,
            img_size = self.img_size,
            patch_size = self.patch_size,
            n_heads = self.n_heads,
            n_blocks = self.n_blocks
            )
    
        self.decoder = SwinDecoder(
            base_dim=self.base_dim,
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size
            )
        
        self.classifier = SwinClassifier(
            self.base_dim, 
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size,
            n_classes = self.n_classes)

    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.classifier(x)
        return x
    
    def predict(self, input, log=False):

        #predict
        output = self(input)

        #output = output.permute(2,1,0)
        if log:
            ic(output.shape)

        # get most probable class
        _,predicted_class = torch.max(output, 0)


        prediction = output

        # transform gardient tensor in normal tensor
        output = output.detach()

        # transform back into numpy array
        output = torch.Tensor.numpy(output.cpu())

        # transpose the axis from b,c,w,h to c,h,w,c
        output =  output.transpose(0,3,2,1)

        # denormalize
        output = output*255
        return output
