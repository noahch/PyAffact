import bob.io.image
import bob.ip.base
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import random
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_tensor

from evaluation.utils import save_input_transform_output_image


class AffactTransformer():
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, config):
        if not config:
            raise Exception("No Config defined")
        self.config = config

    def __call__(self, sample):
        matplotlib.use('Agg')

        im, landmarks, index = sample['image'], sample['landmarks'],  sample['index']

        # Calc bbx
        t_eye_left = np.array((landmarks[0], landmarks[1]))
        t_eye_right = np.array((landmarks[2], landmarks[3]))
        t_mouth_left = np.array((landmarks[4], landmarks[5]))
        t_mouth_right = np.array((landmarks[6], landmarks[7]))

        t_eye = (t_eye_left + t_eye_right) / 2
        t_mouth = (t_mouth_left + t_mouth_right) / 2
        d = np.linalg.norm(t_eye - t_mouth)
        w = h = 5.5 * d
        alpha = np.arctan((t_eye_right[1] - t_eye_left[1]) / (t_eye_right[0] - t_eye_left[0]))

        bbx = [t_eye[0] - 0.5 * w,
               t_eye[1] - 0.45 * h,
               t_eye[0] + 0.5 * w,
               t_eye[1] + 0.55 * h,
               alpha]



        crop_size = [self.config.preprocessing.transformation.crop_size.x, self.config.preprocessing.transformation.crop_size.y]

        # Scale code Version
        scale = min(crop_size[0] / bbx[2], crop_size[1] / bbx[3])

        # Scale paper Version
        scale = min(crop_size[0] / (bbx[2] - bbx[0]), crop_size[1] / (bbx[3] - bbx[1]))


        # TODO: Random bounding box

        angle = bbx[4]
        shift = [0., 0.]

        # Random jitter scale, angle, shift
        if self.config.preprocessing.transformation.scale_jitter.enabled:
            jitter_scale_mean = self.config.preprocessing.transformation.scale_jitter.normal_distribution.mean
            jitter_scale_std = self.config.preprocessing.transformation.scale_jitter.normal_distribution.std
            scale *= np.random.normal(jitter_scale_mean, jitter_scale_std)

        if self.config.preprocessing.transformation.angle_jitter.enabled:
            jitter_angle_mean = self.config.preprocessing.transformation.angle_jitter.normal_distribution.mean
            jitter_angle_std = self.config.preprocessing.transformation.angle_jitter.normal_distribution.std
            angle += np.random.normal(jitter_angle_mean, jitter_angle_std)

        if self.config.preprocessing.transformation.shift_jitter.enabled:
            jitter_shift_mean = self.config.preprocessing.transformation.shift_jitter.normal_distribution.mean
            jitter_shift_std = self.config.preprocessing.transformation.shift_jitter.normal_distribution.std
            shift[0] = int(np.random.normal(jitter_shift_mean, jitter_shift_std) * crop_size[0])
            shift[1] = int(np.random.normal(jitter_shift_mean, jitter_shift_std) * crop_size[1])

        crop_center = [crop_size[0] / 2. + shift[0], crop_size[1] / 2. + shift[1]]
        input_mask = np.ones((im.shape[1], im.shape[2]), dtype=bool)
        out_mask = np.ones((crop_size[0], crop_size[1]), dtype=bool)
        # center = (bbx[1] + bbx[3] / 2., bbx[0] + bbx[2] / 2.)
        center = (bbx[1] + (bbx[3] - bbx[1]) / 2., bbx[0] + (bbx[2] - bbx[0]) / 2.)

        placeholder_out = np.ones((3, self.config.preprocessing.transformation.crop_size.x, self.config.preprocessing.transformation.crop_size.y))
        placeholder_out[placeholder_out > 0] = 0

        geom = bob.ip.base.GeomNorm(angle, scale, crop_size, crop_center)

        for i in range(0, 3):
            in_slice = im[i]
            out_slice = np.ones((crop_size[0], crop_size[1]))
            out_slice = out_slice.astype(np.float)
            x = geom.process(in_slice, input_mask, out_slice, out_mask, center)
            bob.ip.base.extrapolate_mask(out_mask, out_slice)
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
            gaussian = bob.ip.base.Gaussian((sigma, sigma), (int(3. * sigma), int(3. * sigma)))
            gaussian.filter(placeholder_out, placeholder_out)

        # Apply gamma
        if self.config.preprocessing.transformation.gamma.enabled:
            gamma_mean = self.config.preprocessing.transformation.gamma.normal_distribution.mean
            gamma_std = self.config.preprocessing.transformation.gamma.normal_distribution.std
            gamma = 2 ** np.random.normal(gamma_mean, gamma_std)
            placeholder_out = np.minimum(np.maximum(((placeholder_out / 255.0) ** gamma) * 255.0, 0.0), 255.0)

        placeholder_out = np.transpose(placeholder_out, (1, 2, 0))
        placeholder_out = placeholder_out.astype(np.uint8)
        return to_tensor(placeholder_out), bbx