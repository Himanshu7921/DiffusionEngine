from config import config

def load_test_img():
    from PIL import Image
    from torchvision import transforms

    image = Image.open(config["test_img_dir"]).convert("RGB")

    transform = transforms.Compose([
        transforms.ToTensor(),  # PIL image -> Tensor [0, 1]
        transforms.Lambda(lambda x: x * 2.0 - 1.0)  # [0, 1] -> [-1, 1]
    ])

    x_0 = transform(image)
    x_0 = x_0.unsqueeze(0)

    return x_0