# -*- coding: utf-8 -*-
# Yuanzhi Zhu, 2023

import os
import imageio
import argparse
import yaml
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from datetime import datetime

class PixelSampler:
    def __init__(self, path, threshold):
        image = Image.open(path)
        image = image.convert("I")
        image = np.array(image)
        image = image < threshold
        points = np.stack(np.where(image), 1)
        points = (points / image.shape - 0.5) * 2  # => [-1, 1]
        self.image = image
        self.points = points

    @property
    def size(self):
        return np.array(self.image.shape)

    def plot(self, points=None):
        plt.axis('off')
        points = points / 2 + 0.5
        points = np.clip(points, 0.0, 1.)
        points = points * (self.size - 1)
        points = points.astype(int)
        slot = np.zeros(self.size)
        slot[points[:, 0], points[:, 1]] = 1
        plt.imshow(slot, cmap="gray")

def optimal_noise(noisy_latents, particles, sigma):
    ### predict optimal output based on the GT data points: -\nabla U ###
    sigma_2 = sigma ** 2
    def gauss_norm(z, xi): # much faster when pixels are *uncorelated*
        gauss =  torch.exp( -(z - xi)**2 / (2 * sigma_2) ) / ( sigma * torch.sqrt(torch.tensor(2*torch.pi)) )
        log_gauss = torch.log(gauss)
        return log_gauss.sum(dim=list(range(1, len(log_gauss.shape))))
    log_gauss_pdf_list = []
    for i in range(particles.shape[0]):
        log_gauss_pdf_list.append(gauss_norm(noisy_latents, particles[i]))
    log_gauss_pdfs = torch.stack(log_gauss_pdf_list, dim=0).T
    post_softmax = torch.nn.functional.softmax(log_gauss_pdfs, dim=1)
    post_softmax = torch.nan_to_num(post_softmax, nan=0.0)
    weighted_softmax = torch.einsum('bc,ck->bk', post_softmax, particles)
    return 1 / sigma_2 * (weighted_softmax - noisy_latents)

def save_state(particles, ps, work_dir, name):
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)  # adjust subplot parameters
    ps.plot(particles.cpu().numpy())
    plt.savefig(f"{work_dir}/{name}.png", bbox_inches="tight")
    plt.clf()  # clear previous plots

def load_args(path):
    with open(path, "r") as f:
        args = argparse.Namespace(**yaml.load(f, yaml.Loader))
    filename_with_extension = os.path.basename(args.image)
    args.image_name, extension = os.path.splitext(filename_with_extension)
    return args

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=load_args)
    args = parser.parse_args().config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    work_dir = f'output/{args.image_name}_{run_id}'
    os.makedirs(work_dir, exist_ok=True)
    # gt points & particle initialization
    ps = PixelSampler(args.image, args.threshold)
    gt_points = torch.from_numpy(ps.points).to(device)
    save_state(gt_points, ps, work_dir, name=f'ground_truth')
    particles = torch.randn_like(gt_points[:args.n_samples])
    pbar = tqdm.trange(args.n_iters)
    for t in pbar:
        sigma = 0.02 if t < 50 else 0.01 # set current noise level: annealing down
        for _ in range(args.inner_iter): # inner loop
            noisy_particles = (particles) + sigma * torch.randn_like(particles)
            # delta_z = optimal_noise(noisy_particles, gt_points, sigma) - \
            #             optimal_noise(noisy_particles, particles, sigma)
            delta_z = optimal_noise(noisy_particles, gt_points, sigma)
            particles += args.lr * (delta_z * sigma ** 2) + np.sqrt(2*args.lr) * sigma * torch.randn_like(particles)
        if t % args.steps_per_frame == 0 or (t == args.n_iters-1):
            save_state(particles.clone(), ps, work_dir, name=f'{t:06d}_particles')
    # make gif
    images = sorted(Path(work_dir).glob("*_particles.png"))
    images = [imageio.imread(image) for image in images]
    imageio.mimsave(f'{work_dir}/{args.image_name}.gif', images, duration=0.05)

if __name__ == "__main__":
    main()