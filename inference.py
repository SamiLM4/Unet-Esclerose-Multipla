import torch
from PIL import Image
from torchvision import transforms
from model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet()
model.load_state_dict(torch.load("unet_mri_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])


def predict(image: Image):

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)

    mask = (output > 0.5).float()

    return mask.squeeze().cpu().numpy()