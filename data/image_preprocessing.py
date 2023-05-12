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


class PatchifyAugment(Patchify):
    """Convert tensor image into grid of patches, where each path overlaps half of its neighbours. 
    Then applies patch based augmentation.

    Args:
        gray (boolean): defines whether the input tensor is coloured or not
        grid_size (int): defines the output grid size for the patchification
    """

    def __init__(self, gray, grid_size, t1, t2 = None):
        super().__init__(grid_size=grid_size)
        self.gray = gray
        
        # As labeled certain transformations have been written so that they
        # are applied on tensors, this alleviates the need to convert to PIL.Image
        self.transformations = {
            'rotate' : self.Rotate,
            'color'  : self.Color,
            'cutout' : self.Cutout,
            'crop'   : self.Crop
        }

        self.t1 = t1
        self.t2 = t2

        if t2 == '':
            self.transforms = [self.transformations[t1]]
        else:
            self.transforms = [self.transformations[t1], self.transformations[t2]]

    def __call__(self, x):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be patchified and augmented.

        Returns:
            Tensor: Patchified and Augmented Tensor image of shape (grid_size x grid_size x C x patch_size x patch_size)
        """
        # Patchify using parent class
        x = super().__call__(x) 
        self.patch_dim = (self.patch_size, self.patch_size)
        
        # For each patch apply augmentation as in CPC V2
        for patch_row in range(self.grid_size):
            for patch_col in range(self.grid_size):
                patch = x[patch_row][patch_col]

                # Randomly choose two of the 16 (15 if grayscale) transformations from AutoAugment
                for transform in self.transforms:
                    x[patch_row][patch_col] = transform(patch)

        return x

    def __repr__(self):
        return self.__class__.__name__ + f'(grid_size={self.grid_size}, gray={self.gray})'


    def Crop(self, patch):
        patch_PIL = transforms.ToPILImage()(patch)
        left = random.randint(0, 2)
        upper = random.randint(0, 2)
        right = left + self.patch_size - 2
        lower = upper + self.patch_size - 2
        patch_PIL = patch_PIL.crop((left, upper, right, lower))
        patch_PIL = patch_PIL.resize((self.patch_size, self.patch_size))
        return transforms.ToTensor()(patch_PIL)

    def ShearX(self, pil_img):
        level = random.random() * 0.6 - 0.3  # [-0.3,0.3] As in AutoAugment
        return pil_img.transform(self.patch_dim, Image.AFFINE, (1, level, 0, 0, 1, 0))


    def ShearY(self, pil_img):
        level = random.random() * 0.6 - 0.3  # [-0.3,0.3] As in AutoAugment
        return pil_img.transform(self.patch_dim, Image.AFFINE, (1, 0, 0, level, 1, 0))


    def TranslateX(self, patch):
        # Autoaugment does [-150,150] pixels which is eqiuvalent to 45% of 331x331 image
        # 1/3 of patch - 45% seems excessive
        pixels = random.randint(int(-self.patch_size/3), int(self.patch_size/3))
        channels = patch.shape[0]

        # (C, H, W) - columns are dim 2
        if pixels < 0:
            patch = torch.cat((patch[:,:,-pixels:], torch.zeros(channels, self.patch_size, -pixels)), dim=2)
        elif pixels > 0:
            patch = torch.cat((torch.zeros(channels, self.patch_size, pixels), patch[:,:,:self.patch_size-pixels]), dim=2)

        return patch
        #return pil_img.transform(self.patch_dim, Image.AFFINE, (1, 0, pixels, 0, 1, 0))


    def TranslateY(self, patch):
        # Autoaugment does [-150,150] pixels which is eqiuvalent to ~1/2 of 331x331 image
        # 1/3 of patch - 45% seems excessive
        pixels = random.randint(int(-self.patch_size/3), int(self.patch_size/3))
        channels = patch.shape[0]

        # (C, H, W) - rows are dim 1
        if pixels < 0:
            patch = torch.cat((patch[:,-pixels:,:], torch.zeros(channels, -pixels, self.patch_size)), dim=1)
        elif pixels > 0:
            patch = torch.cat((torch.zeros(channels, pixels, self.patch_size), patch[:,:self.patch_size-pixels,:]), dim=1)

        return patch
        #return pil_img.transform(self.patch_dim, Image.AFFINE, (1, 0, pixels, 0, 1, 0))
        

    def Rotate(self, patch):
        patch_PIL = transforms.ToPILImage()(patch)
        degree = random.randint(0, 3) * 90
        patch_PIL = patch_PIL.rotate(degree)
        return transforms.ToTensor()(patch_PIL)


    def Invert(self, patch):
        return 1 - patch


    def Solarize(self, patch):
        threshold = random.random() # [0, 1) - equivalent to [0, 256] as in AutoSegment
        cond = patch >= threshold
        patch[cond] = 1 - patch[cond]
        return patch
        #threshold = random.randint(0, 256)  # [0, 256] as in AutoSegment
        #return PIO.solarize(pil_img, threshold)


    def Posterize(self, patch):
        bits = random.randint(4, 8)  # [4,8] as in AutoSegment
        patch = (patch * 255) // (2 ** (8 - bits)) * (2 ** (8-bits)) / 255
        return patch
        #return PIO.posterize(pil_img, bits)


    def Contrast(self, pil_img):
        level = random.random() * 1.8 + 0.1  # [0.1,1.9] As in AutoAugment
        return PIE.Contrast(pil_img).enhance(level)


    def Color(self, patch):
        patch_PIL = transforms.ToPILImage()(patch)
        strength = 1.0
        patch_PIL = transforms.ColorJitter(brightness=(1-0.8*strength, 1+0.8*strength), contrast=(1-0.8*strength, 1+0.8*strength),
                                           saturation=(1-0.8*strength, 1+0.8*strength), hue=(1-0.2*strength, 1+0.2*strength))(patch_PIL)1
        return transforms.ToTensor()(patch_PIL)


    def Brightness(self, pil_img):
        level = random.random() * 1.8 + 0.1  # [0.1,1.9] As in AutoAugment
        return PIE.Brightness(pil_img).enhance(level)


    def Sharpness(self, pil_img):
        level = random.random() * 1.8 + 0.1  # [0.1,1.9] As in AutoAugment
        return PIE.Sharpness(pil_img).enhance(level)


    def Cutout(self, patch):
        # Autoaugment does [0, 60] pixels which is eqiuvalent to ~1/5th of 331x331 image
        # 1/3 of patch otherwise they are too small
        size = random.randint(1, int(self.patch_size/3))

        # generate top_left crop coordinate
        x_coord = random.randint(0, self.patch_size - size)
        y_coord = random.randint(0, self.patch_size - size)

        patch[:, x_coord:x_coord+size, y_coord:y_coord+size] = 0.5

        return patch


    def SamplePairing(self, patch, other_patch):
        level = random.random() * 0.4 # [0, 0.4] As in AutoAugment
        return patch + level * other_patch


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