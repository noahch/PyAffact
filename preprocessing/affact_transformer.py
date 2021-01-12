"""
Class that handles the AFFACT transformations
"""
import math

import bob.io.image
import bob.ip.base
import matplotlib
import numpy as np
from PIL import Image
import random

from torchvision.transforms.functional import to_tensor


class AffactTransformer():
    """
    Apply AFFACT transformations (scale, rotate, shift, blur, gamma) and temperature to image
    """

    def __init__(self, config):
        """
        Initialization
        :param config: training configuration file
        """
        if not config:
            raise Exception("No Config defined")
        self.config = config

        # Kelvin table for temperature shift
        self.kelvin_table = {
            1000: (255, 56, 0),
            1500: (255, 109, 0),
            2000: (255, 137, 18),
            2500: (255, 161, 72),
            3000: (255, 180, 107),
            3500: (255, 196, 137),
            4000: (255, 209, 163),
            4500: (255, 219, 186),
            5000: (255, 228, 206),
            5500: (255, 236, 224),
            6000: (255, 243, 239),
            6500: (255, 249, 253),
            7000: (245, 243, 255),
            7500: (235, 238, 255),
            8000: (227, 233, 255),
            8500: (220, 229, 255),
            9000: (214, 225, 255),
            9500: (208, 222, 255),
            10000: (204, 219, 255)}


    def __call__(self, sample):
        """
        Transform operations of AFFACT (scale, rotate, shift, blur, gamma) and temperature
        :param sample: dict containing image, landmarks, bounding boxes, index
        :return: torch tensor of transformed image
        """
        matplotlib.use('Agg')

        image, landmarks, bounding_boxes, index = sample['image'], sample['landmarks'], sample['bounding_boxes'], sample['index']

        # Calculate bounding box based on landmarks according to the AFFACT paper
        if self.config.dataset.bounding_box_mode == 0:
            t_eye_left = np.array((landmarks[0], landmarks[1]))
            t_eye_right = np.array((landmarks[2], landmarks[3]))
            t_mouth_left = np.array((landmarks[4], landmarks[5]))
            t_mouth_right = np.array((landmarks[6], landmarks[7]))

            t_eye = (t_eye_left + t_eye_right) / 2
            t_mouth = (t_mouth_left + t_mouth_right) / 2
            d = np.linalg.norm(t_eye - t_mouth)
            w = h = 5.5 * d
            alpha = math.degrees(np.arctan2((t_eye_right[1] - t_eye_left[1]), (t_eye_right[0] - t_eye_left[0])))

            bbx = [t_eye[0] - 0.5 * w,
                   t_eye[1] - 0.45 * h,
                   t_eye[0] + 0.5 * w,
                   t_eye[1] + 0.55 * h,
                   alpha]

        # If no landmarks provided, bounding boxes from dataset or face detector with rotation angle = 0
        else:
            bbx = [
                bounding_boxes[0],
                bounding_boxes[1],
                bounding_boxes[0] + bounding_boxes[2],
                bounding_boxes[1] + bounding_boxes[3],
                0
            ]

        # Define Crop size
        crop_size = [self.config.preprocessing.transformation.crop_size.x, self.config.preprocessing.transformation.crop_size.y]

        # Calculate Scale factor
        scale = min(crop_size[0] / (bbx[2] - bbx[0]), crop_size[1] / (bbx[3] - bbx[1]))

        # Extract rotation angle from bounding box
        rotation_angle = bbx[4]

        # Default shift offset (x,y)
        shift = [0., 0.]

        # Random jitter scale, angle, shift
        if self.config.preprocessing.transformation.scale_jitter.enabled:
            jitter_scale_mean = self.config.preprocessing.transformation.scale_jitter.normal_distribution.mean
            jitter_scale_std = self.config.preprocessing.transformation.scale_jitter.normal_distribution.std
            scale *= 2 ** np.random.normal(jitter_scale_mean, jitter_scale_std)

        if self.config.preprocessing.transformation.angle_jitter.enabled:
            jitter_angle_mean = self.config.preprocessing.transformation.angle_jitter.normal_distribution.mean
            jitter_angle_std = self.config.preprocessing.transformation.angle_jitter.normal_distribution.std
            rotation_angle += np.random.normal(jitter_angle_mean, jitter_angle_std)

        if self.config.preprocessing.transformation.shift_jitter.enabled:
            jitter_shift_mean = self.config.preprocessing.transformation.shift_jitter.normal_distribution.mean
            jitter_shift_std = self.config.preprocessing.transformation.shift_jitter.normal_distribution.std
            shift[0] = int(np.random.normal(jitter_shift_mean, jitter_shift_std) * crop_size[0])
            shift[1] = int(np.random.normal(jitter_shift_mean, jitter_shift_std) * crop_size[1])

        # Calculate crop center
        crop_center = [crop_size[0] / 2. + shift[0], crop_size[1] / 2. + shift[1]]

        # Define an input mask
        input_mask = np.ones((image.shape[1], image.shape[2]), dtype=bool)

        # Define output mask
        out_mask = np.ones((crop_size[0], crop_size[1]), dtype=bool)

        # Calculate Center of bounding box
        center = (bbx[1] + (bbx[3] - bbx[1]) / 2., bbx[0] + (bbx[2] - bbx[0]) / 2.)

        # Empty numpy ndarray (serves as placeholder for new image)
        placeholder_out = np.ones((3, self.config.preprocessing.transformation.crop_size.x, self.config.preprocessing.transformation.crop_size.y))
        placeholder_out[placeholder_out > 0] = 0

        # define geometric normalization
        geom = bob.ip.base.GeomNorm(rotation_angle, scale, crop_size, crop_center)

        # Channel-wise application of geonorm and extrapolation of mask
        for i in range(0, 3):
            in_slice = image[i]
            out_slice = np.ones((crop_size[0], crop_size[1]))
            out_slice = out_slice.astype(np.float)
            x = geom.process(in_slice, input_mask, out_slice, out_mask, center)
            try:
                bob.ip.base.extrapolate_mask(out_mask, out_slice)
            except:
                pass

            # Fill channel
            placeholder_out[i] = out_slice

        # Mirror/Flip Image if randomly drawn number is below probability threshold
        if self.config.preprocessing.transformation.mirror.enabled:
            if random.uniform(0, 1) <= self.config.preprocessing.transformation.mirror.probability:
                placeholder_out = np.flip(placeholder_out, axis=2).copy()

        # Apply Gaussian Blur
        if self.config.preprocessing.transformation.gaussian_blur.enabled:
            sigma_mean = self.config.preprocessing.transformation.gaussian_blur.normal_distribution.mean
            sigma_std = self.config.preprocessing.transformation.gaussian_blur.normal_distribution.std
            sigma = np.random.normal(sigma_mean, sigma_std)
            # Fix: sigma of 0 produces a black image
            if sigma == 0.0:
                sigma = 0.000001
            gaussian = bob.ip.base.Gaussian((sigma, sigma), (int(3. * sigma), int(3. * sigma)))
            gaussian.filter(placeholder_out, placeholder_out)

        # Apply gamma
        if self.config.preprocessing.transformation.gamma.enabled:
            gamma_mean = self.config.preprocessing.transformation.gamma.normal_distribution.mean
            gamma_std = self.config.preprocessing.transformation.gamma.normal_distribution.std
            gamma = 2 ** np.random.normal(gamma_mean, gamma_std)
            placeholder_out = np.minimum(np.maximum(((placeholder_out / 255.0) ** gamma) * 255.0, 0.0), 255.0)

        # Apply Picture Temperature, own contribution and not part of the AFFACT paper
        if self.config.preprocessing.transformation.temperature.enabled:
            r, g, b = self.kelvin_table[random.choice(list(self.kelvin_table.keys()))]
            matrix = (r / 255.0, 0.0, 0.0, 0.0,
                      0.0, g / 255.0, 0.0, 0.0,
                      0.0, 0.0, b / 255.0, 0.0)
            temp_img = Image.fromarray(np.uint8(np.transpose(placeholder_out, (1, 2, 0)))).convert('RGB')
            temp_img = temp_img.convert('RGB', matrix)
            placeholder_out = np.asarray(temp_img)
            placeholder_out = np.transpose(placeholder_out, (2, 0, 1))

        # to create a numpy array of shape H x W x C
        placeholder_out = np.transpose(placeholder_out, (1, 2, 0))

        # convert each pixel to uint8
        placeholder_out = placeholder_out.astype(np.uint8)

        # to_tensor normalizes the numpy array (HxWxC) in the range [0. 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        return to_tensor(placeholder_out)