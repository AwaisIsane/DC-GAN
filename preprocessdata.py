# Create the dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms

image_size = 64
def dataset(dataroot):
    return  dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
