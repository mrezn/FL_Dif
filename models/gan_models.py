import torch
import torch.nn as nn
import torch.nn.functional as F

from fastprogress import progress_bar
import numpy as np
# class CGeneratorA(nn.Module):
#     def __init__(self, nz=100, ngf=64, nc=1, img_size=32, n_cls=10):
#         super(CGeneratorA, self).__init__()

#         self.init_size = img_size//4
#         self.l1 = nn.Sequential(nn.Linear(nz, ngf*self.init_size**2))
#         self.l2 = nn.Sequential(nn.Linear(n_cls, ngf*self.init_size**2))

#         self.conv_blocks0 = nn.Sequential(
#             nn.BatchNorm2d(ngf*2),
#         )
#         self.conv_blocks1 = nn.Sequential(
#             nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
#             nn.BatchNorm2d(ngf*2),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#         self.conv_blocks2 = nn.Sequential(
#             nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
#             nn.BatchNorm2d(ngf),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
#             nn.Tanh(),
#             nn.BatchNorm2d(nc, affine=False)
#         )

#     def forward(self, z, y):
#         out_1 = self.l1(z.view(z.shape[0],-1))
#         out_2 = self.l2(y.view(y.shape[0],-1))
#         out = torch.cat([out_1, out_2], dim=1)
#         out = out.view(out.shape[0], -1, self.init_size, self.init_size)
#         img = self.conv_blocks0(out)
#         img = nn.functional.interpolate(img,scale_factor=2)
#         img = self.conv_blocks1(img)
#         img = nn.functional.interpolate(img,scale_factor=2)
#         img = self.conv_blocks2(img)
#         return img

# class CGeneratorA(nn.Module):
#     def __init__(self, nz=100*32*32, ngf=64, nc=1, img_size=32, n_cls=10):
#         super(CGeneratorA, self).__init__()
#         # Encoder
#         self.enc_conv1 = nn.Conv2d(100, 64, kernel_size=3, padding=1)
#         self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
#         # Decoder
#         self.dec_conv1 = nn.Conv2d(128 + 3+7, 64, kernel_size=3, padding=1)
#         self.dec_conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
#     def forward(self, z, y):
#         # Encoder
#         print(z.shape)
#         print(y.shape)
#         art=input()
#         enc_out1 = torch.relu(self.enc_conv1(z))
#         # print(enc_out1.shape)
#         enc_out2 = torch.relu(self.enc_conv2(self.pool(enc_out1)))
#         # enc_out2 = torch.relu(self.enc_conv2(enc_out1))

#         # Decoder
#         dec_out1 = self.upsample(enc_out2)
#         # print(dec_out1.shape)
#         # print(y.unsqueeze(2).unsqueeze(3).repeat(1, 1, 32, 32).shape)
#         dec_out1 = torch.cat([dec_out1, y.unsqueeze(2).unsqueeze(3).repeat(1, 1, 32, 32)], dim=1)
#         # print(dec_out1.shape)
#         dec_out1 = torch.relu(self.dec_conv1(dec_out1))
#         dec_out2 = torch.sigmoid(self.dec_conv2(dec_out1))

#         return dec_out2
    

class CGeneratorA(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, img_size=32, n_cls=10):
        super(CGeneratorA, self).__init__()

        self.init_size = img_size//4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf*self.init_size**2))
        self.l2 = nn.Sequential(nn.Linear(n_cls, ngf*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False)
        )

    def forward(self, z, y):
        # print(z.shape)
        # print(y.shape)
        # print(y)
        # print(z.view(z.shape[0],-1).shape)
        out_1 = self.l1(z.view(z.shape[0],-1))
        out_2 = self.l2(y.view(y.shape[0],-1))
        out = torch.cat([out_1, out_2], dim=1)
        # print(out_1.shape)
        # print(out_2.shape)
        # print(out.shape)
        # art=input("z y")
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        # print(img.shape)
        # print(type(img))
        # art=input("img shape")
        # print(img)

        # art=input()
        return img


# class ChannelShuffle(nn.Module):
#     def __init__(self,groups):
#         super().__init__()
#         self.groups=groups
#     def forward(self,x):
#         n,c,h,w=x.shape
#         x=x.view(n,self.groups,c//self.groups,h,w) # group
#         x=x.transpose(1,2).contiguous().view(n,-1,h,w) #shuffle
        
#         return x

# class ConvBnSiLu(nn.Module):
#     def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0):
#         super().__init__()
#         self.module=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
#                                   nn.BatchNorm2d(out_channels),
#                                   nn.SiLU(inplace=True))
#     def forward(self,x):
#         return self.module(x)

# class ResidualBottleneck(nn.Module):
#     '''
#     shufflenet_v2 basic unit(https://arxiv.org/pdf/1807.11164.pdf)
#     '''
#     def __init__(self,in_channels,out_channels):
#         super().__init__()

#         self.branch1=nn.Sequential(nn.Conv2d(in_channels//2,in_channels//2,3,1,1,groups=in_channels//2),
#                                     nn.BatchNorm2d(in_channels//2),
#                                     ConvBnSiLu(in_channels//2,out_channels//2,1,1,0))
#         self.branch2=nn.Sequential(ConvBnSiLu(in_channels//2,in_channels//2,1,1,0),
#                                     nn.Conv2d(in_channels//2,in_channels//2,3,1,1,groups=in_channels//2),
#                                     nn.BatchNorm2d(in_channels//2),
#                                     ConvBnSiLu(in_channels//2,out_channels//2,1,1,0))
#         self.channel_shuffle=ChannelShuffle(2)

#     def forward(self,x):
#         x1,x2=x.chunk(2,dim=1)
#         x=torch.cat([self.branch1(x1),self.branch2(x2)],dim=1)
#         x=self.channel_shuffle(x) #shuffle two branches

#         return x

# class ResidualDownsample(nn.Module):
#     '''
#     shufflenet_v2 unit for spatial down sampling(https://arxiv.org/pdf/1807.11164.pdf)
#     '''
#     def __init__(self,in_channels,out_channels):
#         super().__init__()
#         self.branch1=nn.Sequential(nn.Conv2d(in_channels,in_channels,3,2,1,groups=in_channels),
#                                     nn.BatchNorm2d(in_channels),
#                                     ConvBnSiLu(in_channels,out_channels//2,1,1,0))
#         self.branch2=nn.Sequential(ConvBnSiLu(in_channels,out_channels//2,1,1,0),
#                                     nn.Conv2d(out_channels//2,out_channels//2,3,2,1,groups=out_channels//2),
#                                     nn.BatchNorm2d(out_channels//2),
#                                     ConvBnSiLu(out_channels//2,out_channels//2,1,1,0))
#         self.channel_shuffle=ChannelShuffle(2)

#     def forward(self,x):
#         x=torch.cat([self.branch1(x),self.branch2(x)],dim=1)
#         x=self.channel_shuffle(x) #shuffle two branches

#         return x

# class TimeMLP(nn.Module):
#     '''
#     naive introduce timestep information to feature maps with mlp and add shortcut
#     '''
#     def __init__(self,embedding_dim,hidden_dim,out_dim):
#         super().__init__()
#         # print("#"*80)
#         # print(embedding_dim)
#         # print(out_dim)
#         self.mlp=nn.Sequential(nn.Linear(embedding_dim,hidden_dim),
#                                 nn.SiLU(),
#                                nn.Linear(hidden_dim,out_dim))
#         self.act=nn.SiLU()
#     def forward(self,x,t):
#         t_emb=self.mlp(t).unsqueeze(-1).unsqueeze(-1)
#         # print("#"*80)
#         # print(x.shape)
#         # print(t_emb.shape)
#         x=x+t_emb
  
#         return self.act(x)
    
# class EncoderBlock(nn.Module):
#     def __init__(self,in_channels,out_channels,time_embedding_dim):
#         super().__init__()
#         self.conv0=nn.Sequential(*[ResidualBottleneck(in_channels,in_channels) for i in range(3)],
#                                     ResidualBottleneck(in_channels,out_channels//2))

#         self.time_mlp=TimeMLP(embedding_dim=time_embedding_dim,hidden_dim=out_channels,out_dim=out_channels//2)
#         self.conv1=ResidualDownsample(out_channels//2,out_channels)
    
#     def forward(self,x,t=None):
#         x_shortcut=self.conv0(x)
#         if t is not None:
#             x=self.time_mlp(x_shortcut,t)
#         x=self.conv1(x)

#         return [x,x_shortcut]
        
# class DecoderBlock(nn.Module):
#     def __init__(self,in_channels,out_channels,time_embedding_dim):
#         super().__init__()
#         self.upsample=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
#         self.conv0=nn.Sequential(*[ResidualBottleneck(in_channels,in_channels) for i in range(3)],
#                                     ResidualBottleneck(in_channels,in_channels//2))

#         self.time_mlp=TimeMLP(embedding_dim=time_embedding_dim,hidden_dim=in_channels,out_dim=in_channels//2)
#         self.conv1=ResidualBottleneck(in_channels//2,out_channels//2)

#     def forward(self,x,x_shortcut,t=None):
#         x=self.upsample(x)
#         x=torch.cat([x,x_shortcut],dim=1)
#         x=self.conv0(x)
#         if t is not None:
#             x=self.time_mlp(x,t)
#         x=self.conv1(x)

#         return x        

# class CGeneratorA(nn.Module):
#     '''
#     simple unet design without attention
#     '''
#     def __init__(self, nz=100, ngf=64, nc=1, img_size=32, n_cls=10):
#         super(CGeneratorA, self).__init__()
#         timesteps=1000
#         time_embedding_dim=128
#         in_channels=1
#         num_class=10
#         out_channels=3
#         base_dim=32
#         dim_mults=[2,4,8,16]
#         assert isinstance(dim_mults,(list,tuple))
#         assert base_dim%2==0 

#         channels=self._cal_channels(base_dim,dim_mults)

#         self.init_conv=ConvBnSiLu(in_channels,base_dim,3,1,1)
#         self.init_conv_lable=ConvBnSiLu(num_class,base_dim,3,1,1)

#         self.time_embedding=nn.Embedding(timesteps,time_embedding_dim)
#         # print([c[0] for c in channels] )
#         # print([c[1]for c in channels])
#         self.encoder_blocks=nn.ModuleList([EncoderBlock(c[0],c[1],time_embedding_dim) for c in channels])
#         self.decoder_blocks=nn.ModuleList([DecoderBlock(c[1],c[0],time_embedding_dim) for c in channels[::-1]])
    
#         self.mid_block=nn.Sequential(*[ResidualBottleneck(channels[-1][1],channels[-1][1]) for i in range(2)],
#                                         ResidualBottleneck(channels[-1][1],channels[-1][1]//2))

#         self.final_conv=nn.Conv2d(in_channels=channels[0][0]//2,out_channels=out_channels,kernel_size=1)

#     def forward(self,x,y,t=None):
#         x=self.init_conv(x)
#         y=self.init_conv_lable(y)
#         # print(y.shape)
#         # print(x.shape)
#         # art=input("x y")
#         y_expanded = y.expand(-1, -1, -1, 32) 
#         x = torch.cat((x, y_expanded), dim=2)

#         if t is not None:
#             t=self.time_embedding(t)
#         encoder_shortcuts=[]
#         # print(x.shape)
#         # print(type(x))
#         # art=input("x y")
#         for encoder_block in self.encoder_blocks:
#             x,x_shortcut=encoder_block(x,t)
#             encoder_shortcuts.append(x_shortcut)
#         x=self.mid_block(x)
#         encoder_shortcuts.reverse()
#         for decoder_block,shortcut in zip(self.decoder_blocks,encoder_shortcuts):
#             x=decoder_block(x,shortcut,t)
#         x=self.final_conv(x)

#         return x

#     def _cal_channels(self,base_dim,dim_mults):
#         dims=[base_dim*x for x in dim_mults]
#         dims.insert(0,base_dim)
#         channels=[]
#         for i in range(len(dims)-1):
#             channels.append((dims[i],dims[i+1])) # in_channel, out_channel

#         return channels

# def one_param(m):
#     "get model first parameter"
#     return next(iter(m.parameters()))
# class SelfAttention(nn.Module):
#     def __init__(self, channels):
#         super(SelfAttention, self).__init__()
#         self.channels = channels        
#         self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
#         self.ln = nn.LayerNorm([channels])
#         self.ff_self = nn.Sequential(
#             nn.LayerNorm([channels]),
#             nn.Linear(channels, channels),
#             nn.GELU(),
#             nn.Linear(channels, channels),
#         )

#     def forward(self, x):
#         size = x.shape[-1]
#         x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
#         x_ln = self.ln(x)
#         attention_value, _ = self.mha(x_ln, x_ln, x_ln)
#         attention_value = attention_value + x
#         attention_value = self.ff_self(attention_value) + attention_value
#         return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
#         super().__init__()
#         self.residual = residual
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.GroupNorm(1, mid_channels),
#             nn.GELU(),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.GroupNorm(1, out_channels),
#         )

#     def forward(self, x):
#         if self.residual:
#             return F.gelu(x + self.double_conv(x))
#         else:
#             return self.double_conv(x)


# class Down(nn.Module):
#     def __init__(self, in_channels, out_channels, emb_dim=256):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, in_channels, residual=True),
#             DoubleConv(in_channels, out_channels),
#         )

#         self.emb_layer = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(
#                 emb_dim,
#                 out_channels
#             ),
#         )

#     def forward(self, x, t):
#         x = self.maxpool_conv(x)
#         emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
#         return x + emb


# class Up(nn.Module):
#     def __init__(self, in_channels, out_channels, emb_dim=256):
#         super().__init__()

#         self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         self.conv = nn.Sequential(
#             DoubleConv(in_channels, in_channels, residual=True),
#             DoubleConv(in_channels, out_channels, in_channels // 2),
#         )

#         self.emb_layer = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(
#                 emb_dim,
#                 out_channels
#             ),
#         )

#     def forward(self, x, skip_x, t):
#         x = self.up(x)
#         x = torch.cat([skip_x, x], dim=1)
#         x = self.conv(x)
#         emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
#         return x + emb

# class UNet(nn.Module):
#     def __init__(self, c_in=3, c_out=3, time_dim=256, remove_deep_conv=False):
#         super().__init__()
#         self.time_dim = time_dim
#         self.remove_deep_conv = remove_deep_conv
#         self.inc = DoubleConv(c_in, 64)
#         self.down1 = Down(64, 128)
#         self.sa1 = SelfAttention(128)
#         self.down2 = Down(128, 256)
#         self.sa2 = SelfAttention(256)
#         self.down3 = Down(256, 256)
#         self.sa3 = SelfAttention(256)


#         if remove_deep_conv:
#             self.bot1 = DoubleConv(256, 256)
#             self.bot3 = DoubleConv(256, 256)
#         else:
#             self.bot1 = DoubleConv(256, 512)
#             self.bot2 = DoubleConv(512, 512)
#             self.bot3 = DoubleConv(512, 256)

#         self.up1 = Up(512, 128)
#         self.sa4 = SelfAttention(128)
#         self.up2 = Up(256, 64)
#         self.sa5 = SelfAttention(64)
#         self.up3 = Up(128, 64)
#         self.sa6 = SelfAttention(64)
#         self.outc = nn.Conv2d(64, c_out, kernel_size=1)

#     def pos_encoding(self, t, channels):
#         inv_freq = 1.0 / (
#             10000
#             ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels)
#         )
#         pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
#         pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
#         pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
#         return pos_enc

#     def unet_forwad(self, x, t):
#         x1 = self.inc(x)
#         x2 = self.down1(x1, t)
#         x2 = self.sa1(x2)
#         x3 = self.down2(x2, t)
#         x3 = self.sa2(x3)
#         x4 = self.down3(x3, t)
#         x4 = self.sa3(x4)

#         x4 = self.bot1(x4)
#         if not self.remove_deep_conv:
#             x4 = self.bot2(x4)
#         x4 = self.bot3(x4)

#         x = self.up1(x4, x3, t)
#         x = self.sa4(x)
#         x = self.up2(x, x2, t)
#         x = self.sa5(x)
#         x = self.up3(x, x1, t)
#         x = self.sa6(x)
#         output = self.outc(x)
#         return output
    
#     def forward(self, x, t):
#         t = t.unsqueeze(-1)
#         t = self.pos_encoding(t, self.time_dim)
#         return self.unet_forwad(x, t)

    
# class CGeneratorA(UNet):
#     def __init__(self, nz=100, ngf=64, nc=1, img_size=32, n_cls=10):
#         super(CGeneratorA, self).__init__()
#         # num_classes=10
#         if n_cls is not None:
#             self.label_emb = nn.Embedding(n_cls, 256)
#     def forward(self, x, t, y=None):
#         t = t.unsqueeze(-1)
#         t = self.pos_encoding(t, self.time_dim)

#         if y is not None:
#             t += self.label_emb(y)

#         return self.unet_forwad(x, t)
    

# def sample(y , model_fu):
#     labels=creat_label_from_onehot(y)
#     n = len(labels)
#     #     logging.info(f"Sampling {n} new images....")
#     model_fu.eval()
#     with torch.inference_mode():
#         x = torch.randn((n, CGeneratorA.c_inn, CGeneratorA.img_sizee, CGeneratorA.img_sizee)).to(CGeneratorA.devicee)
#         for i in progress_bar(reversed(range(1, CGeneratorA.noise_stepss)), total=CGeneratorA.noise_stepss-1, leave=False):
#             print(i)
#             # art=input()
#             t = (torch.ones(n) * i).long().to(CGeneratorA.devicee)
#             print(t.shape)
#             print(x.shape)
#             print(f"labels : {labels}")
#             # print(f"labels _ 2 : {labels.item()}")
#             predicted_noise = model_fu(x, t, labels)
#             print(f"predicted_noise tpye: {type(predicted_noise)}")
#             print(f"predicted_noise shape: {(predicted_noise.shape)}")
#             if CGeneratorA.cfg_scale > 0:
#                 uncond_predicted_noise = CGeneratorA.model_fu(x, t, None)
#                 predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, CGeneratorA.cfg_scale)
#             alpha = CGeneratorA.alpha_t[t][:, None, None, None]
#             alpha_hat = CGeneratorA.alpha_hat_t[t][:, None, None, None]
#             beta = CGeneratorA.beta_t[t][:, None, None, None]
#             if i > 1:
#                 noise = torch.randn_like(x)
#             else:
#                 noise = torch.zeros_like(x)
#             x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
#     x = (x.clamp(-1, 1) + 1) / 2
#     x = (x * 255).type(torch.uint8)
#     print(f"x shape : {x.shape}")
#     return x
    
# def prepare_noise_schedule(beta_start,noise_steps,beta_end):
#     return torch.linspace(beta_start, beta_end, noise_steps)

# def creat_label_from_onehot(y_onehot):
#     batch_sizee=y_onehot.shape[0]
#     data_len_classes=y_onehot.shape[1]
#     labels=torch.ones(batch_sizee).long().cpu()
#     for i in range(batch_sizee):
#         for j in range(data_len_classes):
#             if y_onehot[i][j] !=0:
#                 labels[i]=j
#                 break
#     return(labels)

# class CGeneratorA(nn.Module):
#     def __init__(self, nz=100, ngf=64, nc=1, img_size=32, n_cls=10):
#         super(CGeneratorA, self).__init__()

        # num_classes=10
        # self.c_inn=3
        # self.c_outt=3
        # self.devicee="cpu"
        # self.img_sizee=32
        # self.noise_stepss=100
        # beta_start=1e-4
        # beta_end=0.02
        # use_ema=False
        # self.cfg_scale=3
        # cifar_labels = "airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck".split(",")
        # self.beta_t = prepare_noise_schedule(beta_start,self.noise_stepss,beta_end).to(self.devicee)
        # self.alpha_t= 1. - self.beta_t
        # self.alpha_hat_t = torch.cumprod(self.alpha_t, dim=0)
        # self.model_fu = UNet_conditional(self.c_inn, self.c_outt, num_classes=num_classes).to(self.devicee)








# def sample(use_ema, labels,img_size, cfg_scale,alpha_t, alpha_hat_t , beta_t,device,noise_steps,model,c_in):
#     n = len(labels)
# #     logging.info(f"Sampling {n} new images....")
#     model.eval()
#     with torch.inference_mode():
#         x = torch.randn((n, c_in, img_size, img_size)).to(device)
#         for i in progress_bar(reversed(range(1, noise_steps)), total=noise_steps-1, leave=False):
#             print(i)
#             # art=input()
#             t = (torch.ones(n) * i).long().to(device)
#             print(t.shape)
#             print(x.shape)
#             print(f"labels : {labels}")
#             # print(f"labels _ 2 : {labels.item()}")
#             predicted_noise = model(x, t, labels)
#             print(f"predicted_noise tpye: {type(predicted_noise)}")
#             print(f"predicted_noise shape: {(predicted_noise.shape)}")
#             if cfg_scale > 0:
#                 uncond_predicted_noise = model(x, t, None)
#                 predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
#             alpha = alpha_t[t][:, None, None, None]
#             alpha_hat = alpha_hat_t[t][:, None, None, None]
#             beta = beta_t[t][:, None, None, None]
#             if i > 1:
#                 noise = torch.randn_like(x)
#             else:
#                 noise = torch.zeros_like(x)
#             x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
#     x = (x.clamp(-1, 1) + 1) / 2
#     x = (x * 255).type(torch.uint8)
#     print(f"x shape : {x.shape}")
#     return x