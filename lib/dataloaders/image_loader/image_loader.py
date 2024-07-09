from os import listdir
from os.path import join, isfile, splitext, basename

import cv2
from torch.utils.data import Dataset


def calculate_image(image_size, origin_image_size):
    aspect_ratio = origin_image_size[1] / origin_image_size[0]
    return aspect_ratio, (int(image_size * aspect_ratio) - int(image_size * aspect_ratio) % 16, image_size)


class ImageLoader(Dataset):
    def __init__(
            self,
            origin_image_path,
            gt_image_path,
            mode='train',
            transforms=None,
            support_types: [str] = None,
            gt_format: str = None,
    ):
        """Initializes image paths and preprocessing module."""
        if support_types is None:
            support_types = ['jpg', 'png', 'jpeg', 'bmp', 'tif', 'tiff', 'JPG', 'PNG', 'JPEG', 'BMP', 'TIF', 'TIFF']
        support_types = set(support_types)
        self.origin_image_path = origin_image_path
        self.ground_truth_path = gt_image_path
        self.image_paths = [join(origin_image_path, f) for f in listdir(origin_image_path)
                            if isfile(join(origin_image_path, f)) and splitext(f)[1][1:] in support_types]
        self.mode = mode
        self.transforms = transforms
        self.gt_format = gt_format
        print(f"Dataset Type: {self.mode}, image count: {len(self.image_paths)}")

    def get_gt_file_name(self, origin_image_name: str, extension: str) -> str:
        try:
            return self.gt_format.format(origin_image_name)
        except:
            return origin_image_name + "_segmentation.png"

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        origin_image_name, extension = splitext(basename(image_path))
        filename = self.get_gt_file_name(origin_image_name, extension)
        GT_path = join(self.ground_truth_path, filename)

        image = cv2.imread(image_path, -1)
        label = cv2.imread(GT_path, -1)

        try:
            if label.shape[-1] == 3:
                label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        except:
            print("Error in reading the label image: ", GT_path)
            raise

        label[label < 128] = 0
        label[label >= 128] = 1

        image, label = self.transforms(image, label)
        return image, label, origin_image_name

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)
