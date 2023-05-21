import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import PIL.ImageOps as PIO
import PIL.ImageEnhance as PIE
import random

class Patchify(object):     
    """Convert tensor image into grid of patches, where each path overlaps half of its neighbours

    Args:
        grid_size (int): defines the output grid size for the patchification
    """
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.patch_size = None

    def __call__(self, x):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be patchified.

        Returns:
            Tensor: Patchified Tensor image of shape (grid_size x grid_size x C x patch_size x patch_size)
        """

        # Calculate the size of the patches
        if self.patch_size is None:         
            # Patchifying requires a square input image
            if x.shape[1] != x.shape[2]:
                raise Exception("Patchifying requires a square input image")
            
            patch_size = float(x.shape[2] / (self.grid_size + 1) * 2)

            # Not all grid sizes are compatable, ensure that patch_size is a whole number
            if patch_size.is_integer():
                self.patch_size = int(patch_size)
            else:
                raise Exception("The specified grid size did not fit the image")

        # Input x = (channels, img_size, img_size)
        # Patchify to (grid_size, grid_size, channels, patch_size, patch_size)
        x = (
            x.unfold(1, self.patch_size, self.patch_size // 2)
            .unfold(2, self.patch_size, self.patch_size // 2)
            .permute(1, 2, 0, 3, 4)
            .contiguous()
        )

        return x

    def __repr__(self):
        return self.__class__.__name__ + '(grid_size={0})'.format(self.grid_size)



class PrePatchAugNormalizeReshape(object):
    """
    Converts a tensor (grid_size x grid_size x C x patch_size x patch_size) to (C x  grid_size**2 x patch_size**2)
    """
    def __call__(self, img):
        # Move C to start
        img = img.permute(2, 0, 1, 3, 4)

        # Combine dimnensions
        img = img.view(img.shape[0], img.shape[1] * img.shape[2], img.shape[3] * img.shape[4]) 

        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


class PostPatchAugNormalizeReshape(object):
    """
    Converts a tensor (C x  grid_size**2 x patch_size**2) to (grid_size x grid_size x C x patch_size x patch_size)
    """
    def __call__(self, img):
        # Calcualte grid and patch size
        grid_size = int(img.shape[1] ** 0.5)
        patch_size = int(img.shape[2] ** 0.5)

        # Get rid of grid_size**2 and patch_size**2
        img = img.view(img.shape[0], grid_size, grid_size, patch_size, patch_size) 

        # Move C to dim 2
        img = img.permute(1, 2, 0, 3, 4)

        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


class PatchAugNormalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given the input shape of (grid_size x grid_size x C x patch_size x patch_size) 
    it performs reshaping before and after the normalization

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Args:
            tensor (Tensor): Tensor image of size (grid_size x grid_size x C x patch_size x patch_size) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        img = PrePatchAugNormalizeReshape()(img)
        img = transforms.Normalize(mean=self.mean, std=self.std)(img)
        img = PostPatchAugNormalizeReshape()(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'