
import segmentation_models_pytorch as smp
from src.SwinUnetTransformer.genericSwinUnet import GenericSwinUnet




def generate_model(config):
    if config['General']['model_type'] == 'unet':        
        new_model = smp.Unet('xception', encoder_weights='imagenet', in_channels=config['Train']['input_bands'],
            encoder_depth=4, decoder_channels=[128, 64, 32, 16], classes=config['Dataset']['class_n'])
    elif config['General']['model_type'] == 'deeplab': # use this one        
        
        new_model = smp.DeepLabV3Plus('tu-xception41', encoder_weights='imagenet', 
                                    in_channels=config['Train']['input_bands'],
                                    classes=config['Dataset']['class_n'])
    elif config['General']['model_type'] == 'resunet':
        new_model = smp.Unet('resnet34', encoder_weights='imagenet', in_channels=config['Train']['input_bands'], 
                                    classes=config['Dataset']['class_n'])
    elif config['General']['model_type'] == 'resunet-fromScratch':        
        new_model = smp.Unet('resnet34', encoder_weights=None, in_channels=config['Train']['input_bands'],
                                    classes=config['Dataset']['class_n'])

        
    elif config['General']['model_type'] == 'swinTrans':
        
        config_dict = {}
        config_dict['n_classes'] = config['Dataset']['class_n']
        config_dict['img_size'] = config['Dataset']['patch_size']

        config_dict['base_dim'] = 96
        config_dict['window_size'] = [4, 4]
        config_dict['shift_size'] = [2, 2]
        config_dict['patch_size'] = [4, 4]
        config_dict['n_heads'] = [3, 6, 12, 24]
        config_dict['n_blocks'] = [2, 2, 2, 2]
        new_model = GenericSwinUnet(config_dict)
        new_model.prepare_model(config['General']['input_bands'])

    else:
        raise f"Unknow model type {config['General']['model_type']}"

    return new_model