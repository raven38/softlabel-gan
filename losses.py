import torch
import torch.nn.functional as F

# DCGAN loss


def loss_dcgan_dis(dis_fake, dis_real):
    L1 = torch.mean(F.softplus(-dis_real))
    L2 = torch.mean(F.softplus(dis_fake))
    return L1, L2


def loss_dcgan_gen(dis_fake):
    loss = torch.mean(F.softplus(-dis_fake))
    return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real, weight_real=None):
    if weight_real is None:
        loss_real = torch.mean(F.relu(1. - dis_real))
        loss_fake = torch.mean(F.relu(1. + dis_fake))
        return loss_real, loss_fake
    else:
        real = F.relu(1. - dis_real)
        weighted_real = torch.mean(real * weight_real)
        loss_real = (torch.mean(real).detach() / weighted_real.detach()).detach() * weighted_real
        loss_fake = torch.mean(F.relu(1. + dis_fake))
        return loss_real, loss_fake
        


def loss_hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss


def loss_hinge_cvdis(dis_fake, dis_real):
    loss_real = 0.
    for each in dis_real:
        loss_real += torch.mean(F.relu(1. - each))
    loss_fake = 0.
    for each in dis_fake:
        loss_fake += torch.mean(F.relu(1. + each))
    return loss_real, loss_fake


def loss_hinge_cvgen(dis_fake):
    loss = 0.
    for each in dis_fake:
        loss += -torch.mean(each)
    return loss

# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis

cvgenerator_loss = loss_hinge_cvgen
cvdiscriminator_loss = loss_hinge_cvdis
