import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset_brats import BraTSDataset
from diffusion3d import GaussianDiffusion3D
from unet3d_conditional import ConditionalUNet3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
ds = BraTSDataset("data", split="val", k_slices=5)
cond2d, target3d = ds[0]

cond2d = cond2d.unsqueeze(0).to(device)

# Rebuild model
model = ConditionalUNet3D(
    in_channels=1,
    out_channels=1,
    k_slices=5,
    base_dim=16,
    levels=3,
    groupnorm_groups=8,
    depth_size=ds.crop_size,
).to(device)

diffusion = GaussianDiffusion3D(model=model, timesteps=1000)

ckpt = torch.load("checkpoints_brats/best.pt", map_location=device)
model.load_state_dict(ckpt["ema"])
model.eval()

with torch.no_grad():
    sample = diffusion.sample(
        model,
        cond2d,
        shape=(1, 1, ds.crop_size, ds.crop_size, ds.crop_size),
    )

sample = sample[0, 0].cpu().numpy()
target = target3d[0].numpy()

# Plot conditioning slices
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i in range(5):
    axes[0, i].imshow(cond2d[0, i].cpu(), cmap="gray")
    axes[0, i].axis("off")
    axes[0, i].set_title("Cond")

# Compare center slices
d = sample.shape[0]
zs = [d//4, d//2, 3*d//4]

for i, z in enumerate(zs):
    axes[1, i].imshow(sample[z], cmap="gray")
    axes[1, i].set_title("Generated")
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()


from mpl_toolkits.mplot3d import Axes3D

threshold = np.percentile(sample, 85)
coords = np.argwhere(sample > threshold)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(coords[:,2], coords[:,1], coords[:,0], s=1, c="blue")
plt.show()


mse = np.mean((sample - target)**2)
print("MSE:", mse)
