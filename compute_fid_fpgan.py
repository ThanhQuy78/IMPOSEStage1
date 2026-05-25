"""
Compute FID between FPGAN-generated fingerprints and real SD302a images.
Uses robust matrix sqrt with epsilon correction for domain-specific images.
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class ImageFolderFlat(Dataset):
    """Load all PNGs from a directory, convert grayscale → RGB, resize to 299."""

    def __init__(self, folder):
        self.paths = sorted(glob.glob(os.path.join(folder,"**","*.png"), recursive=True))
        if not self.paths:
            raise ValueError(f"No PNG images found in {folder}")
        self.transform = transforms.Compose([
            transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),  # [0,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def get_inception_model(device):
    """Load InceptionV3 with pool3 features (2048-dim)."""
    from torchvision.models import inception_v3
    model = inception_v3(pretrained=True, transform_input=False)
    # We need features before the final FC layer
    model.fc = torch.nn.Identity()
    model.eval()
    return model.to(device)


def compute_features(dataloader, model, device):
    """Extract 2048-dim Inception features for all images."""
    features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            # Normalize to ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            batch = batch.to(device)
            batch = (batch - mean) / std
            feat = model(batch)
            features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute FID using eigenvalue decomposition (robust for rank-deficient covariances)."""
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Regularize covariances
    offset = np.eye(sigma1.shape[0]) * eps
    sigma1 = sigma1 + offset
    sigma2 = sigma2 + offset

    # Compute sqrt(sigma1 @ sigma2) via eigendecomposition of sigma1
    # FID = ||mu1-mu2||^2 + Tr(sigma1) + Tr(sigma2) - 2*Tr(sqrtm(sigma1 @ sigma2))
    # Use: sqrtm(A @ B) = sqrt_A @ sqrtm(sqrt_A^T @ B @ sqrt_A) @ sqrt_A^{-1} ... too complex
    # Instead, directly compute trace via: Tr(sqrtm(sigma1 @ sigma2)) = sum(sqrt(eigenvalues(sigma1 @ sigma2)))
    # But sigma1 @ sigma2 is not symmetric. Use the identity:
    # Tr(sqrtm(A @ B)) = Tr(sqrtm(sqrt_A @ B @ sqrt_A)) where sqrt_A = sqrtm(A)
    # And sqrt_A @ B @ sqrt_A IS symmetric.

    # Eigendecompose sigma1
    s1_eigvals, s1_eigvecs = np.linalg.eigh(sigma1)
    s1_eigvals = np.maximum(s1_eigvals, 0)
    sqrt_s1 = s1_eigvecs * np.sqrt(s1_eigvals)[None, :] @ s1_eigvecs.T

    # Compute S = sqrt_s1 @ sigma2 @ sqrt_s1 (symmetric)
    S = sqrt_s1 @ sigma2 @ sqrt_s1

    # Eigendecompose S to get Tr(sqrtm(S))
    s_eigvals = np.linalg.eigvalsh(S)
    s_eigvals = np.maximum(s_eigvals, 0)
    tr_covmean = np.sum(np.sqrt(s_eigvals))

    fid = diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    # Remove eps contribution: we added eps*I to both, so Tr(s1)+Tr(s2) each gained eps*dim
    dim = sigma1.shape[0]
    fid = fid - 2 * eps * dim  # correct for the eps added to traces
    return float(max(fid, 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str, required=True)
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb_project", type=str, default="datadream-sd302a-fingerprint")
    parser.add_argument("--wandb_entity", type=str, default="RL_team_BTML")
    parser.add_argument("--wandb_run_name", type=str, default="fid-fpgan-vs-real")
    args = parser.parse_args()

    device = torch.device(args.device)

    gen_ds = ImageFolderFlat(args.gen_dir)
    real_ds = ImageFolderFlat(args.real_dir)
    print(f"Generated: {len(gen_ds)} images from {args.gen_dir}")
    print(f"Real:      {len(real_ds)} images from {args.real_dir}")

    gen_loader = DataLoader(gen_ds, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    real_loader = DataLoader(real_ds, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    print("\nLoading InceptionV3...")
    model = get_inception_model(device)

    print("Extracting generated image features...")
    gen_feats = compute_features(gen_loader, model, device)
    print(f"  Feature shape: {gen_feats.shape}")

    print("Extracting real image features...")
    real_feats = compute_features(real_loader, model, device)
    print(f"  Feature shape: {real_feats.shape}")

    mu_gen, sigma_gen = gen_feats.mean(axis=0), np.cov(gen_feats, rowvar=False)
    mu_real, sigma_real = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)

    print("\nComputing FID...")
    fid_score = frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

    print(f"\n{'='*50}")
    print(f"FID Score: {fid_score:.4f}")
    print(f"{'='*50}")
    print(f"  Generated: {args.gen_dir} ({len(gen_ds)} images)")
    print(f"  Real:      {args.real_dir} ({len(real_ds)} images)")

    # Log to wandb
    try:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "gen_dir": args.gen_dir,
                "real_dir": args.real_dir,
                "n_generated": len(gen_ds),
                "n_real": len(real_ds),
                "method": "fpgan",
            },
        )
        wandb.log({"FID": fid_score})
        wandb.finish()
        print(f"\nFID logged to wandb: {args.wandb_project} / {args.wandb_run_name}")
    except Exception as e:
        print(f"\nWarning: Could not log to wandb: {e}")


if __name__ == "__main__":
    main()
