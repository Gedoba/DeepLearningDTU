from tqdm.notebook import tqdm
import torch
from model import MainModel
from visualize import visualize


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count


def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()

    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}


def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)


def save_model(path, model, epoch, loss_meter_dict):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict_G': model.opt_G.state_dict(),
        'optimizer_state_dict_D': model.opt_D.state_dict(),
        'losses': loss_meter_dict
    }, path)


def load_model(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.opt_D.load_state_dict(checkpoint['optimizer_state_dict_D'])
    model.opt_G.load_state_dict(checkpoint['optimizer_state_dict_G'])
    epoch = checkpoint['epoch']
    model.epoch = epoch
    print(model.epoch)
    loss_meter_dict = checkpoint['losses']
    return model, loss_meter_dict


def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")


def train_model(model, train_dl, val_dl, color_space, epochs, display_every=200, loss_meter_dict=None, save_path=None):
    # getting a batch for visualizing the model output after fixed intrvals
    vis_data = next(iter(val_dl))
    starting_epoch = model.epoch
    for e in range(epochs):
        # function returing a dictionary of objects to
        # log the losses of the complete network
        loss_meter_dict = create_loss_meters() if loss_meter_dict is None else loss_meter_dict
        i = 0
        for data in tqdm(train_dl):
            model.setup_input(data)
            model.optimize()
            # function updating the log objects
            update_losses(model, loss_meter_dict,
                          count=data['known_channel'].size(0))
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {model.epoch + 1}/{epochs + starting_epoch}")
                print(f"Iteration {i}/{len(train_dl)}")
                # function to print out the losses
                log_results(loss_meter_dict)
                # function displaying the model's outputs
                visualize(model, vis_data, color_space, save=False)
        model.epoch += 1
        if save_path is not None:
            save_model(save_path, model, model.epoch, loss_meter_dict)
