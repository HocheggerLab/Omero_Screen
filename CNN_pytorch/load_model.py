import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import torchvision.transforms as transforms
from CNN_pytorch.CNN_model import CNN
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
import ast
import skimage
def load_checkpoint(checkpoint,model,optimizer):
    print('=> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()


def trained_model():
    # Set the device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    model = CNN(in_channels=3,num_classes=2,).to(device=device)
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    load_checkpoint(torch.load('/Users/haoranyue/PycharmProjects/Omero_Screen_2/CNN_pytorch/CNN_checkpoint.pth.tar'),model,optimizer)
    model=model.to(device=device)
    return model.eval()


def data_transform():
    transform = A.Compose([
        A.Resize(32, 32, ),
        # transforms.ToPILImage(),
        # T.Resize((32, 32),
        # T.InterpolationMode.BICUBIC),
        # transforms.ToTensor(),
        ToTensorV2(),
    ])
    return transform


def data_tra():
    transform = A.Compose(
        [
            A.Resize(32, 32, ),
            # # A.CenterCrop(60, 60, ),
            # # A.Resize(32, 32),
            A.Rotate(limit=20, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.VerticalFlip(p=0.6),
            A.VerticalFlip(p=0.4),
            A.OneOf(
                [
                    # A.Blur(blur_limit=3, p=0.8),
                    # A.Blur(blur_limit=3, p=0.5),
                    # A.ColorJitter(p=0.6),

                ], p=1.0
            ),
            ToTensorV2(),

        ]
    )

# def predict_images(image_list):
#     model=trained_model()
#     transform=data_transform()
#     # Convert the list of PIL images to PyTorch tensors
#
#     tensor_list = [transform(Image.open(image)) for image in image_list]
#     # Stack the tensors into a single batch
#     batch = torch.stack(tensor_list)
#     # Pass the batch through the model
#     with torch.no_grad():
#         outputs = model(batch)
#     # Get the predicted class probabilities
#     probs = torch.nn.functional.softmax(outputs, dim=1)
#     return probs.numpy()

if __name__ == "__main__":
    df=np.load('/Users/haoranyue/Desktop/mm.npy',allow_pickle=True)
    # img =skimage.io.imread(df['cell_data'].tolist()[0])

    print(df[0][3])
    # arr_str = np.fromstring(df['cell_data'].iloc[0])
    # arr = np.array(eval(df['cell_data'].iloc[0])).astype('float32')
    plt.imshow(df[0][3])
    plt.show()
    # print(float(df['training_data'].tolist()[0]).shape)
    # Use ast.literal_eval to safely evaluate the string
    # array_obj = ast.literal_eval(df['cell_data'].tolist()[0][0])

    # Convert the Python object to a numpy array
    # array = np.array(array_obj)

    # print(array)
    # plt.show()

    # predict_images(image_list=)
