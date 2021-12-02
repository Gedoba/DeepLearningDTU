from tqdm.notebook import tqdm
import torch
from visualize import visualize, plot_metrics
from IPython.display import clear_output, display
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18, resnet34
from torchvision.models.vgg import vgg16_bn
from fastai.vision.models.unet import DynamicUnet
import numpy as np
class AverageMeter:
    def __init__(self):
        self.reset()
        self.values = []

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count
        self.values.append(self.avg)


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


def update_val_losses(val_loss_dict, iter_loss_dict, count):
    for loss_name, loss_meter in val_loss_dict.items():
        loss = iter_loss_dict[loss_name]
        loss_meter.update(loss.item(), count)


def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")

def save_model(path, model, loss_meter_dict, train_loss_meter_dict):
    torch.save({
        'epoch': model.epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict_G': model.opt_G.state_dict(),
        'optimizer_state_dict_D': model.opt_D.state_dict(),
        'losses': loss_meter_dict,
        'train_losses': train_loss_meter_dict
    }, path)


def build_backbone_unet(n_input=1, n_output=2, size=256, backbone_name='resnet18'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = resnet18
    if backbone_name == 'resnet18':
        backbone = resnet18
    elif backbone_name == 'resnet34':
        backbone = resnet34
    elif backbone_name == 'vgg16_bn':
        backbone = vgg16_bn

    body = create_body(backbone, pretrained=True, n_in=n_input, cut=-2) #backbones: resnet18, resnet34, vgg16_bn
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G

def load_model(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.opt_D.load_state_dict(checkpoint['optimizer_state_dict_D'])
    model.opt_G.load_state_dict(checkpoint['optimizer_state_dict_G'])
    epoch = checkpoint['epoch']
    model.epoch = epoch
    print(model.epoch)
    train_loss_meter_dict = None
    loss_meter_dict = checkpoint['losses']
    if 'train_losses' in checkpoint:
        train_loss_meter_dict = checkpoint['train_losses']
    return model, loss_meter_dict, train_loss_meter_dict





def train_model(model, train_dl, val_dl, color_space, epochs, display_every=200, loss_meter_dict=None, save_path=None, val_loss_meter_dict=None):
    # getting a batch for visualizing the model output after fixed intrvals
    vis_data = next(iter(val_dl))
    starting_epoch = model.epoch
    iters = [] if starting_epoch == 0 else list(range(1, starting_epoch * len(train_dl) + 1))
    iter_count = 0 if starting_epoch == 0 else starting_epoch * len(train_dl) - 1
    early_stopping = EarlyStopping(path = save_path, verbose=True)
    for e in range(epochs):
        # function returing a dictionary of objects to
        # log the losses of the complete network
        loss_meter_dict = create_loss_meters() if loss_meter_dict is None else loss_meter_dict
        i = 0
        model.train()
        pbar = tqdm(train_dl)
        for data in pbar:
            model.setup_input(data)
            model.optimize()
            # function updating the log objects
            update_losses(model, loss_meter_dict,
                          count=data['known_channel'].size(0))
            i += 1
            iter_count += 1
            iters.append(iter_count)
            if i % display_every == 0:
                print(f"\nEpoch {model.epoch + 1}/{epochs + starting_epoch}")
                print(f"Iteration {i}/{len(train_dl)}")
                # function to print out the losses
                display(pbar.container)
                plot_metrics(iters, loss_meter_dict)
                # function displaying the model's outputs
                visualize(model, vis_data, color_space, save=False)
                clear_output(wait=True)

        model.eval()
        val_loss_meter_dict = create_loss_meters() if val_loss_meter_dict is None else val_loss_meter_dict
        val_iter_loss_meter_dict = {}
        val_batches = 0
        for val_data in val_dl:
            model.setup_input(val_data)
            tmp_losses = model.model_eval()
            for loss_name, loss_meter in tmp_losses.items():
                if loss_name not in val_iter_loss_meter_dict:
                    val_iter_loss_meter_dict[loss_name] = 0
                val_iter_loss_meter_dict[loss_name] += loss_meter
            val_batches += 1

        for loss_name, loss_meter in val_iter_loss_meter_dict.items():
            val_iter_loss_meter_dict[loss_name] = val_iter_loss_meter_dict[loss_name] / val_batches

        update_val_losses(val_loss_meter_dict, val_iter_loss_meter_dict, val_batches)
        val_epochs = list(range(1, starting_epoch + e + 2))
        plot_metrics(val_epochs, val_loss_meter_dict, True)

        model.train()

        model.epoch += 1
        early_stopping(model, val_loss_meter_dict, loss_meter_dict)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break


def pretrain_generator(net_G, train_dl, opt, criterion, epochs, device):
    for e in range(epochs):
        loss_meter = AverageMeter()
        for data in tqdm(train_dl):
            known_channel, unknown_channels = data['known_channel'].to(device), data['unknown_channels'].to(device)
            preds = net_G(known_channel)
            loss = criterion(preds, unknown_channels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            loss_meter.update(loss.item(), known_channel.size(0))
            
        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter.avg:.5f}")



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, path = None, patience=5, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, model, loss_meter_dict, train_loss_meter_dict):
        val_loss = loss_meter_dict['loss_G'].avg
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, loss_meter_dict, train_loss_meter_dict)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, loss_meter_dict, train_loss_meter_dict)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, loss_meter_dict, train_loss_meter_dict):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).{"  Saving model ..." if self.path is not None else ""}')
        if self.path is not None:
            save_model(self.path, model, loss_meter_dict, train_loss_meter_dict)
        self.val_loss_min = val_loss