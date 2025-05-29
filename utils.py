
import torch
import Config


def save_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    print("=> saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=>loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=Config.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.load_state_dict(checkpoint["optimizer"])

    for params in optimizer.param_groups:
        params["lr"] = lr
