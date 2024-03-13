import torch.nn as nn
from torchvision.models import *
from torchvision.utils import *
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
import kornia as ko
from torch.distributions import Categorical


class AdaptiveEren:
    '''
    Estimates Threshold using the adaptive method given in Eq.4 of main paper 
    '''
    def __init__(self, param, device):
        super().__init__()
        self.steps = 100 # Number of iterations
        self.device = device
        self.alpha = 0.001
        self.edge_operator = ko.filters.Sobel(normalized=True)
        self.params = param

        if self.params['dataset'] == 'cifar10' or self.params['dataset'] == 'cifar10c':
            self.mean_entropy = 19.56
        elif self.params['dataset'] == 'cifar100' or self.params['dataset'] == 'cifar100c':
            self.mean_entropy = 19.57
        elif self.params['dataset'] == 'tiny_imagenet' or self.params['dataset'] == 'tiny_imagenetc':
            self.mean_entropy = 23.60
        elif self.params['dataset'] == 'imagenet_9' or self.params['dataset'] == 'imagenet_9c':
            self.mean_entropy = 30.83
        else:
            raise ValueError('dataset not supported')

    def edge_maps_func(self, x):
        # expecting only one channel
        edge_maps = self.edge_operator(x)
        # rescale to 0-1
        edge_maps = (edge_maps - edge_maps.min()) / (edge_maps.max() - edge_maps.min())
        return edge_maps

    def edge_loss(self, corr_images):
        edge_corr = self.edge_maps_func(corr_images)
        net_loss = Categorical(probs=abs(edge_corr.reshape(edge_corr.shape[0],-1)), validate_args=False).entropy().mean()
        return net_loss
    
    def __call__(self, corr_images, extra_arg=None):
        corr_images = corr_images.clone().detach().to(self.device)

        corr_r_channel = corr_images[:, 0:1, :, :]
        corr_g_channel = corr_images[:, 1:2, :, :]
        corr_b_channel = corr_images[:, 2:3, :, :]

        adv_images_r = corr_r_channel.clone().detach()
        adv_images_g = corr_g_channel.clone().detach()
        adv_images_b = corr_b_channel.clone().detach()
        
        with torch.no_grad():
            initial_corruption_entropy = self.edge_loss(corr_r_channel) + self.edge_loss(corr_g_channel) + self.edge_loss(corr_b_channel)
            custom_threshold = 2*self.mean_entropy - initial_corruption_entropy # can be fixed for T_25, ie threshold as 25th percentile
            print('using threshold: ', custom_threshold)

        for s in range(self.steps):
            adv_images_r.requires_grad = True
            adv_images_g.requires_grad = True
            adv_images_b.requires_grad = True

            cost1 = self.edge_loss(adv_images_r)
            cost2 = self.edge_loss(adv_images_g)
            cost3 = self.edge_loss(adv_images_b)
            cost = cost1 + cost2 + cost3

            if cost < custom_threshold:
                print('using threshold: ', custom_threshold)
                print('breaking at step: ', s)
                break

            grad_r = torch.autograd.grad(
                cost, adv_images_r, retain_graph=False, create_graph=False)[0]
            grad_g = torch.autograd.grad(
                cost, adv_images_g, retain_graph=False, create_graph=False)[0]
            grad_b = torch.autograd.grad(
                cost, adv_images_b, retain_graph=False, create_graph=False)[0]

            adv_images_r = adv_images_r.detach() - self.alpha * grad_r.sign()
            adv_images_g = adv_images_g.detach() - self.alpha * grad_g.sign()
            adv_images_b = adv_images_b.detach() - self.alpha * grad_b.sign()

        adv_images_r = torch.clamp(
            adv_images_r, min=0, max=1).detach()
        adv_images_g = torch.clamp(
            adv_images_g, min=0, max=1).detach()
        adv_images_b = torch.clamp(
            adv_images_b, min=0, max=1).detach()
        adv_images = torch.cat(
            (adv_images_r, adv_images_g, adv_images_b), dim=1)

        return adv_images
