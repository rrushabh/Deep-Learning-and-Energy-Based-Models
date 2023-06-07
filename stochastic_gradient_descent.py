import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import VGG13_BN_Weights, vgg13_bn
from tqdm import tqdm

DEVICE = "cpu"  # "cuda"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def save_img(image, path):
    # Push to CPU, convert from (1, 3, H, W) into (H, W, 3)
    image = image[0].permute(1, 2, 0)
    image = image.clamp(min=0, max=1)
    image = (image * 255).cpu().detach().numpy().astype(np.uint8)
    # opencv expects BGR (and not RGB) format
    cv.imwrite(path, image[:, :, ::-1])


def main():
    model = vgg13_bn(VGG13_BN_Weights.IMAGENET1K_V1).to(DEVICE)
    print(model)
    for label in [0, 12, 954]:
        image = torch.randn(1, 224, 224, 3).to(DEVICE)
        image = (image * 8 + 128) / 255  # background color = 128,128,128
        image = image.permute(0, 3, 1, 2)
        image.requires_grad_()
        image = gradient_descent(image, model, lambda tensor: tensor[0, label].mean(),)
        save_img(image, f"./img_{label}.jpg")
        out = model(image)
        print(f"ANSWER_FOR_LABEL_{label}: {out.softmax(1)[0, label].item()}")


def normalize_and_jitter(img, step=32):
    # data augmentation and normalization,
    # convnets expect values to be mean 0 and std 1
    dx, dy = np.random.randint(-step, step - 1, 2)
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(
        img.roll(dx, -1).roll(dy, -2)
    )


def gradient_descent(input, model, loss, iterations=256):
    learning_rate = 0
    weight_decay = 0.005
    GaussianBlurrer = transforms.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 3))

    for epoch in tqdm(range(iterations), desc="Optimising image"):

        # Set model in inference mode
        model.eval()

        # Implement gradient descent at multiple scales, scaling up every so often
        if epoch%25 == 0:
            learning_rate += 0.015

        # Calculate logit
        # Blur the image at each iteration to reduce high frequency noise
        l = model(normalize_and_jitter(GaussianBlurrer(input)))

        # Calculate the energy
        # Implement weight decay
        F = -(loss(l) + weight_decay*(input**2).sum())

        F.backward()

        # Gradient Descent step
        with torch.no_grad():
            # Added gradient clipping to prevent the exploding gradient
            input.grad.clamp(min=-20, max=20)
            # Blur the gradients at each iteration
            input -= learning_rate * GaussianBlurrer(input.grad)
        
        input.grad.zero_()

        # Clamp the pixel values between 0 and 1
        input.clamp(min=0, max=1)

    return input


def forward_and_return_activation(model, input, module):
    """
    Given a module in the middle of the model,
    return the intermediate activations.
    """
    features = []

    def hook(model, input, output):
        features.append(output)

    handle = module.register_forward_hook(hook)
    model(input)
    handle.remove()

    return features[0]


if __name__ == "__main__":
    main()