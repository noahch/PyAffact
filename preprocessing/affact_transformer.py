import bob.io.image
import bob.ip.base
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import random

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

        im, bbx, index = sample['image'], sample['bounding_box'],  sample['index']

        crop_size = [self.config.preprocessing.transformation.crop_size.x, self.config.preprocessing.transformation.crop_size.y]
        scale = min(crop_size[0] / bbx[2], crop_size[1] / bbx[3])

        # TODO: Random bounding box

        # TODO: ASk.. bbx[4] = angle.. not in given data.. default 0?
        angle = 0
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
        center = (bbx[1] + bbx[3] / 2., bbx[0] + bbx[2] / 2.)
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
                placeholder_out = np.flip(placeholder_out, axis=2)

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

        # Save every X picture to validate preprocessing
        if self.config.preprocessing.transformation.save_transformation_image.enabled:
            if index % self.config.preprocessing.transformation.save_transformation_image.frequency == 0:
                plt.ion()
                plt.clf()
                plt.figure(1)
                plt.subplot(121)
                plt.imshow(Image.fromarray(np.transpose(im, (1, 2, 0)), 'RGB'))
                plt.plot([bbx[0], bbx[0], bbx[0] + bbx[2], bbx[0] + bbx[2]], [bbx[1], bbx[1] + bbx[3], bbx[1], bbx[1] + bbx[3]],
                         'rx', ms=10, mew=3)
                plt.subplot(122)
                out_slice = np.transpose(placeholder_out, (1, 2, 0))
                out_slice = out_slice.astype(np.uint8)
                plt.imshow(Image.fromarray(out_slice, 'RGB'))
                plt.savefig(self.config.basic.result_directory + '/{}.jpg'.format(index))

        return torch.from_numpy(placeholder_out).float()