from main import Resnet
from main import ResidualBlock  # import the model modules
import os
import torch
from torchvision.transforms import transforms
from PIL import Image, ImageDraw, ImageFont

model_path = r"G:\zyj\0421resnet\show\model_save\epoch197+loss0.367+acc0.935+lr0.00049.pth"
test_root = r"G:\zyj\0421resnet\dataset\single_prediction"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# to get the path of each test images
img_path = []
for roots, _, files in os.walk(test_root):
    for file in files:
        img_path.append(os.path.join(roots, file))


def predict():
    model = Resnet(ResidualBlock, [3, 4, 6, 3]).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # no need to update parameters in the testing part
    with torch.no_grad():
        # does not read the images with ImageFolder,
        # since there are only five testing pictures.
        for raw_img in img_path:
            raw_img = Image.open(raw_img)
            draw = ImageDraw.Draw(raw_img)

            # pre-process with the images to fit the network (data argumentation not included)
            normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
            transform = transforms.Compose([
                transforms.Resize(300),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),  # Image -> after ToTensor: (W, H)->(C, H, W)
                normalize
            ])
            img = transform(raw_img)

            # add one dimension, so that the dimension format for the network to read in is (N, C, H, W).
            # here N = 1
            img = img.unsqueeze(0)
            img = img.to(device)

            output = model(img)

            # 'predicted' represents the predicted label,
            # 'score' represents the probability of this predicted label
            score, predicted = torch.max(output, 1)

            # change the label (0,1) to the real classification name (cat or dog)
            if predicted.item() == 0:
                predicted = "cat"
            else:
                predicted = "dog"

            print("this picture might be: ", predicted, ", score: ", score.item())

            text = predicted + str(score.item())
            font = ImageFont.truetype(r"C:\Windows\Fonts\Arial.ttf", size=20)
            draw.text((0,0), text, (255,0,0), font=font)  # write some text on  pictures
            raw_img.show()


if __name__ == '__main__':
    predict()
