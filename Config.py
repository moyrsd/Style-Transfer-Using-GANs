import torchvision.transforms as tt
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
train_dir = "horse2zebra/train"
test_dir = "horse2zebra/test"
batch_size = 4
learning_rate = 0.0002
lambda_identity = 0.0
lambda_cycle = 10
num_workers = 2
epochs = 30
load_model = False
save_model = True
checkpoint_gen_horse = "genh.pth.tar"
checkpoint_gen_zebra = "genz.pth.tar"
checkpoint_dis_horse = "dish.pth.tar"
checkpoint_dis_zebra = "disz.pth.tar"
image_size = 256

transforms = tt.Compose([tt.Resize(image_size),
                         tt.CenterCrop(image_size),
                         tt.ToTensor(),
                         tt.Normalize(mean=[0.5, 0.5, 0.5],
                                      std=[0.5, 0.5, 0.5]),
                         tt.RandomHorizontalFlip(p=0.5)])
