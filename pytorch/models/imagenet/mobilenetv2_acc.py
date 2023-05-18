import torch
import os
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
os.environ["CUDA_VISIBLE_DEVICES"]="1"

model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    model.to('cuda')

def clean_accuracy(data_loader, model):
    clean_dataset = []; correct = 0; total = 0; i = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            i = i + 1
            if i % 500 == 0:
                acc = (i, 100 * correct / total)
                print('INFO: Accuracy of the network on the test images: %d, %.2f %%' % acc)
        
mean = [0.485, 0.456, 0.406] # https://pytorch.org/vision/stable/models.html#wide-resnet
std  = [0.229, 0.224, 0.225]
normalization = [transforms.Normalize(mean=mean, std=std)]
transform_list = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()] + normalization)
IMAGENET_PATH  = "/data/ImageNet1k" # imagenet 2012
TEST_PATH      = os.path.join(IMAGENET_PATH, 'val') # contains subfolders 50k samples

test_loader   = torch.utils.data.DataLoader(datasets.ImageFolder(TEST_PATH, transform_list), batch_size=1, shuffle=False, num_workers=10, pin_memory=True)

clean_accuracy(test_loader, model)


