import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors
from tqdm import tqdm

# flake8: noqa
# mypy: ignore-errors


class tonemap:
    def __init__(self):
        pass

    def process(self, img):
        return img

    def inv_process(self, img):
        return img


# Log correction
class log_tonemap(tonemap):
    # Constructor
    # Base of log
    # Scale of tonemapped
    # Offset
    def __init__(self, base, scale=1, offset=1):
        self.base = base
        self.scale = scale
        self.offset = offset

    def process(self, img):
        tonemapped = (np.log(img + self.offset) / np.log(self.base)) * self.scale
        return tonemapped

    def inv_process(self, img):
        inverse_tonemapped = np.power(self.base, (img) / self.scale) - self.offset
        return inverse_tonemapped


class log_tonemap_clip(tonemap):
    # Constructor
    # Base of log
    # Scale of tonemapped
    # Offset
    def __init__(self, base, scale=1, offset=1):
        self.base = base
        self.scale = scale
        self.offset = offset

    def process(self, img):
        tonemapped = np.clip((np.log(img * self.scale + self.offset) / np.log(self.base)), 0, 2) - 1
        return tonemapped

    def inv_process(self, img):
        inverse_tonemapped = (np.power(self.base, (img + 1)) - self.offset) / self.scale
        return inverse_tonemapped


# Gamma Tonemap
class gamma_tonemap(tonemap):
    def __init__(
        self,
        gamma,
    ):
        self.gamma = gamma

    def process(self, img):
        tonemapped = np.power(img, 1 / self.gamma)
        return tonemapped

    def inv_process(self, img):
        inverse_tonemapped = np.power(img, self.gamma)
        return inverse_tonemapped


class linear_clip(tonemap):
    def __init__(self, scale, mean):
        self.scale = scale
        self.mean = mean

    def process(self, img):
        tonemapped = np.clip((img - self.mean) / self.scale, -1, 1)
        return tonemapped

    def inv_process(self, img):
        inverse_tonemapped = img * self.scale + self.mean
        return inverse_tonemapped


def make_tonemap_HDR(opt):
    if opt.mode == "luminance":
        res_tonemap = log_tonemap_clip(10, 1.0, 1.0)
    else:  # temperature
        res_tonemap = linear_clip(5000.0, 5000.0)
    return res_tonemap


class LDRfromHDR:
    def __init__(
        self, tonemap="none", orig_scale=False, clip=True, quantization=0, color_jitter=0, noise=0
    ):
        self.tonemap_str, val = tonemap
        if tonemap[0] == "gamma":
            self.tonemap = gamma_tonemap(val)
        elif tonemap[0] == "log10":
            self.tonemap = log_tonemap(val)
        else:
            print("Warning: No tonemap specified, using linear")

        self.clip = clip
        self.orig_scale = orig_scale
        self.bits = quantization
        self.jitter = color_jitter
        self.noise = noise

        self.wbModel = None

    def process(self, HDR):
        LDR, normalized_scale = self.rescale(HDR)
        LDR = self.apply_clip(LDR)
        LDR = self.apply_scale(LDR, normalized_scale)
        LDR = self.apply_tonemap(LDR)
        LDR = self.colorJitter(LDR)
        LDR = self.gaussianNoise(LDR)
        LDR = self.quantize(LDR)
        LDR = self.apply_white_balance(LDR)
        return LDR, normalized_scale

    def rescale(self, img, percentile=90, max_mapping=0.8):
        r_percentile = np.percentile(img, percentile)
        alpha = max_mapping / (r_percentile + 1e-10)

        img_reexposed = img * alpha

        normalized_scale = normalizeScale(1 / alpha)

        return img_reexposed, normalized_scale

    def rescaleAlpha(self, img, percentile=90, max_mapping=0.8):
        r_percentile = np.percentile(img, percentile)
        alpha = max_mapping / (r_percentile + 1e-10)

        return alpha

    def apply_clip(self, img):
        if self.clip:
            img = np.clip(img, 0, 1)
        return img

    def apply_scale(self, img, scale):
        if self.orig_scale:
            scale = unNormalizeScale(scale)
            img = img * scale
        return img

    def apply_tonemap(self, img):
        if self.tonemap_str == "none":
            return img
        gammaed = self.tonemap.process(img)
        return gammaed

    def quantize(self, img):
        if self.bits == 0:
            return img
        max_val = np.power(2, self.bits)
        img = img * max_val
        img = np.floor(img)
        img = img / max_val
        return img

    def colorJitter(self, img):
        if self.jitter == 0:
            return img
        hsv = colors.rgb_to_hsv(img)
        hue_offset = np.random.normal(0, self.jitter, 1)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_offset) % 1.0
        rgb = colors.hsv_to_rgb(hsv)
        return rgb

    def gaussianNoise(self, img):
        if self.noise == 0:
            return img
        noise_amount = np.random.uniform(0, self.noise, 1)
        noise_img = np.random.normal(0, noise_amount, img.shape)
        img = img + noise_img
        img = np.clip(img, 0, 1).astype(np.float32)
        return img

    def apply_white_balance(self, img):
        if self.wbModel is None:
            return img
        img = self.wbModel.correctImage(img)
        return img.copy()


def make_LDRfromHDR(opt):
    LDR_from_HDR = LDRfromHDR(
        opt.tonemap_LDR, opt.orig_scale, opt.clip, opt.quantization, opt.color_jitter, opt.noise
    )
    return LDR_from_HDR


def torchnormalizeEV(EV, mean=5.12, scale=6, clip=True):
    # Normalize based on the computed distribution between -1 1
    EV -= mean
    EV = EV / scale

    if clip:
        EV = torch.clip(EV, min=-1, max=1)

    return EV


def torchnormalizeEV0(EV, mean=5.12, scale=6, clip=True):
    # Normalize based on the computed distribution between 0 1
    EV -= mean
    EV = EV / scale

    if clip:
        EV = torch.clip(EV, min=-1, max=1)

    EV += 0.5
    EV = EV / 2

    return EV


def normalizeScale(x, scale=4):
    x = np.log10(x + 1)

    x = x / (scale / 2)
    x = x - 1

    return x


def unNormalizeScale(x, scale=4):
    x = x + 1
    x = x * (scale / 2)

    x = np.power(10, x) - 1

    return x


def normalizeIlluminance(x, scale=5):
    x = np.log10(x + 1)

    x = x / (scale / 2)
    x = x - 1

    return x


def unNormalizeIlluminance(x, scale=5):
    x = x + 1
    x = x * (scale / 2)

    x = np.power(10, x) - 1

    return x


def main(args):
    processor = LDRfromHDR(
        # tonemap=("log10", 10),
        tonemap=("gamma", args.gamma),
        orig_scale=False,
        clip=True,
        quantization=0,
        color_jitter=0,
        noise=0,
    )

    img_list = list(os.listdir(args.hdr_dir))
    img_list = [f for f in img_list if f.endswith(args.extension)]
    img_list = [f for f in img_list if not f.startswith("._")]

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for fname in tqdm(img_list):
        fname_out = ".".join(fname.split(".")[:-1])
        out = os.path.join(args.out_dir, f"{fname_out}.jpg")
        if os.path.exists(out) and not args.overwrite:
            continue

        fpath = os.path.join(args.hdr_dir, fname)
        img = cv2.imread(fpath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ldr, scale = processor.process(img)

        ldr = (ldr * 255).astype(np.uint8)
        ldr = cv2.cvtColor(ldr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out, ldr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdr_dir", type=str, default="hdr")
    parser.add_argument("--out_dir", type=str, default="ldr")
    parser.add_argument("--extension", type=str, default=".exr")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gamma", type=float, default=2)
    args = parser.parse_args()

    main(args)
