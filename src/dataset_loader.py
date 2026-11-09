from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CelebADataset(Dataset):
    def __init__(self, root_dir, image_size=64):
        self.root_dir = Path(root_dir)
        extensions = {".jpg", ".jpeg", ".png"}

        self.image_paths = sorted([
            p for p in self.root_dir.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ])

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No images found in {self.root_dir}"
            )

        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image