from models import resnet
# from models import simple_models
# from models import vgg
from models import gan_models
from diff.ddpm_conditional import Diffusion


def choose_model(model_name, **kwargs):
    if model_name == 'resnet18':
        return resnet.resnet18(**kwargs)
    else:
        raise ValueError('Wrong model name.')

def choose_g_model(model_name, **kwargs):
    if model_name == 'CGeneratorA':
        return Diffusion(noise_steps=500 , img_size=32)#najafi
        # return gan_models.CGeneratorA(**kwargs)
    else:
        raise ValueError('Wrong model name.')
