import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
# from augmentation_utils import generate_gmm_image
# from cddgm_inference import apply_cddgm
from sklearn.metrics import accuracy_score, confusion_matrix

class BrainDataset(Dataset):
    def __init__(self, images_dir, labels_dir, patch_size=(128, 128, 128)):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.patch_size = patch_size
        self.samples = sorted([f for f in os.listdir(images_dir) if f.endswith('_0000.nii.gz')])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_filename = self.samples[idx]
        label_filename = image_filename.replace('_0000', '')

        image = nib.load(os.path.join(self.images_dir, image_filename)).get_fdata().astype(np.float32)
        label = nib.load(os.path.join(self.labels_dir, label_filename)).get_fdata().astype(np.int64)

        # normalize
        image = (image - image.mean()) / (image.std() + 1e-5)

        # pad
        pad_dims = [(0, max(0, p - s)) for s, p in zip(image.shape, self.patch_size)]
        image = np.pad(image, pad_dims, mode='constant')
        label = np.pad(label, pad_dims, mode='constant')

        # crop center patch
        start = [s // 2 - p // 2 for s, p in zip(image.shape, self.patch_size)]
        end = [start[i] + self.patch_size[i] for i in range(3)]
        image_patch = image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        label_patch = label[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        # convert to tensor
        image_patch = torch.from_numpy(image_patch).unsqueeze(0) # shape: [1, D, H, W]
        label_patch = torch.from_numpy(label_patch) # shape: [D, H, W]
        return image_patch, label_patch

class AugBrainDataset(BrainDataset):
    def __init__(self, images_dir, labels_dir, patch_size=(128, 128, 128),
                 apply_gmm=True, apply_cddgm=True):
        super().__init__(images_dir, labels_dir, patch_size)
        self.apply_gmm = apply_gmm
        self.apply_cddgm = apply_cddgm

    def __getitem__(self, idx):
        image_patch, label_patch = super().__getitem__(idx)
        # if self.apply_gmm:
        #     image_patch = generate_gmm_image(image_patch.squeeze(0).numpy())
        #     image_patch = torch.from_numpy(image_patch).unsqueeze(0)
        # if self.apply_cddgm:
        #     image_patch = apply_cddgm(image_patch.squeeze(0).numpy())
        #     image_patch = torch.from_numpy(image_patch).unsqueeze(0)
        return image_patch, label_patch

############## BLOCKS ##############
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + identity)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = ResidualBlock(in_channels, out_channels)
        self.down = nn.Conv3d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.block(x)
        return self.down(x), x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ResidualBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

############## NETWORK ##############
class CrossAgeMultiTaskNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, tumor_classes=4, is_adult_output_dim=1):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, base_channels)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4)

        self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 8)

        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels)

        self.tumor_head = nn.Conv3d(base_channels, tumor_classes, kernel_size=1)

        self.is_adult_pool = nn.AdaptiveAvgPool3d(1)
        self.is_adult_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels, base_channels),
            nn.ReLU(),
            nn.Linear(base_channels, is_adult_output_dim)
        )

    def forward(self, x):
        x1_p, x1 = self.enc1(x)
        x2_p, x2 = self.enc2(x1_p)
        x3_p, x3 = self.enc3(x2_p)

        x = self.bottleneck(x3_p)

        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)

        tumor_out = self.tumor_head(x)

        is_adult_feat = self.is_adult_pool(x)
        is_adult_out = self.is_adult_head(is_adult_feat)
        return tumor_out, is_adult_out

############## LOSS FUNCTION ##############
def dice_loss(pred, target, smooth=1e-5):
    pred = torch.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
    intersection = (pred * target_onehot).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target_onehot.sum(dim=(2, 3, 4))
    return 1 - ((2 * intersection + smooth) / (union + smooth)).mean()

class MultiTaskLoss(nn.Module):
    def __init__(self, alpha_tumor=1.0, alpha_is_adult=0.0):
        super().__init__()
        self.alpha_tumor = alpha_tumor
        self.alpha_is_adult = alpha_is_adult
        self.ce_tumor = nn.CrossEntropyLoss()
        self.bce_is_adult = nn.BCEWithLogitsLoss()

    def forward(self, tumor_pred, is_adult_pred, tumor_gt, is_adult_gt):
        tumor_loss = self.ce_tumor(tumor_pred, tumor_gt) + dice_loss(tumor_pred, tumor_gt)

        if is_adult_pred is None or is_adult_gt is None:
            return self.alpha_tumor * tumor_loss, tumor_loss, torch.tensor(0.0)

        is_adult_gt = is_adult_gt.float().unsqueeze(1)
        is_adult_loss = self.bce_is_adult(is_adult_pred, is_adult_gt)
        total_loss = self.alpha_tumor * tumor_loss + self.alpha_is_adult * is_adult_loss

        return total_loss, tumor_loss, is_adult_loss

############## OPTIMIZER, SCHEDULER ##############
def get_optimizer_and_scheduler(model, lr=1e-3, weight_decay=1e-5):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    return optimizer, scheduler

############## FOR EVALUATION ##############
def dice_score(pred, target, num_classes=4):
    pred = torch.argmax(pred, dim=1)
    pred = pred.view(-1)
    target = target.view(-1)
    scores = []
    for cls in range(1, num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item()
        if union == 0:
            scores.append(1.0)
        else:
            scores.append(2.0 * intersection / (union + 1e-5))
    return np.mean(scores)

def evaluate_model(model, dataloader, is_adult_label, device, num_classes=4):
    model.eval()
    dice_scores = []
    age_labels = []
    age_preds = []

    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            tumor_out, is_adult_out = model(x)

            # tumor segmentation dice
            dice = dice_score(tumor_out, y, num_classes)
            dice_scores.append(dice)

            # binary classification
            is_adult_gt = torch.full((x.size(0),), is_adult_label, dtype=torch.float32, device=device)
            is_adult_prob = torch.sigmoid(is_adult_out)
            is_adult_pred = (is_adult_prob > 0.5).long().squeeze()

            age_labels.extend(is_adult_gt.cpu().numpy())
            age_preds.extend(is_adult_pred.cpu().numpy())

    return {
        "dice": np.mean(dice_scores),
        "accuracy": accuracy_score(age_labels, age_preds),
        "confusion_matrix": confusion_matrix(age_labels, age_preds)
    }

############## PRETRAINING ##############
# settings
path_to_brats_train_imgs = "/path/to/adult/imagesTr"
path_to_brats_train_labels = "/path/to/adult/labelsTr"
path_to_ped_train_imgs = "/path/to/ped/imagesTr"
path_to_ped_train_labels = "/path/to/ped/labelsTr"
path_to_brats_test_imgs = "/path/to/adult/imagesTs"
path_to_brats_test_labels = "/path/to/adult/labelsTs"
path_to_ped_test_imgs = "/path/to/ped/imagesTs"
path_to_ped_test_labels = "/path/to/ped/labelsTs"

pre_dataset = BrainDataset(images_dir=path_to_brats_train_imgs, labels_dir=path_to_brats_train_labels)
pre_dataloader = DataLoader(pre_dataset, batch_size=1, shuffle=True, num_workers=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CrossAgeMultiTaskNet(tumor_classes=4).to(device)

loss_fn = MultiTaskLoss(alpha_tumor=1.0, alpha_is_adult=0.0)  
optimizer, scheduler = get_optimizer_and_scheduler(model)

# learning loop
for epoch in range(25):  # 25 epochs
    model.train()
    epoch_loss = 0
    for x, y in tqdm(pre_dataloader):
        tumor_pred, _ = model(x)
        loss, tumor_loss = loss_fn(tumor_pred, None, y, None)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step(epoch_loss)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

torch.save(model.state_dict(), "pretrained_brats.pth")
print("Pretrained model saved to pretrained_brats.pth")

############## FINE TUNING ##############
# use pretrained model for multi-task learning
brats_dataset = BrainDataset(path_to_brats_train_imgs, path_to_brats_train_labels)
pediatric_dataset = BrainDataset(path_to_ped_train_imgs, path_to_ped_train_labels)

brats_loader = DataLoader(brats_dataset, batch_size=2, shuffle=True, num_workers=1)
pediatric_loader = DataLoader(pediatric_dataset, batch_size=2, shuffle=True, num_workers=1)

pretrained_model_path = "/path/to/pretrained_brats.pth"

model = CrossAgeMultiTaskNet(tumor_classes=4).to(device)
if os.path.exists(pretrained_model_path):
    model.load_state_dict(torch.load(pretrained_model_path))

loss_fn = MultiTaskLoss(alpha_tumor=10.0, alpha_is_adult=0.9)
optimizer, scheduler = get_optimizer_and_scheduler(model, lr=0.001)

for epoch in range(50):
    model.train()
    epoch_loss = 0.0

    for (x1, y1), (x2, y2) in zip(brats_loader, pediatric_loader):
        # merge child and adult data
        x = torch.cat([x1, x2], dim=0).to(device)
        y = torch.cat([y1, y2], dim=0).to(device)

        # generate is_adult labels: 1 for adult, 0 for pediatric
        is_adult = torch.cat([
            torch.ones(len(x1), dtype=torch.float32),
            torch.zeros(len(x2), dtype=torch.float32)
        ], dim=0).to(device)

        # forward pass and caculate loss
        tumor_pred, is_adult_pred = model(x)
        loss, loss_tumor, loss_is_adult = loss_fn(tumor_pred, is_adult_pred, y, is_adult)

        # backprop and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # update schedular
    scheduler.step(epoch_loss)
    print(f"[Epoch {epoch+1}] Total Loss: {epoch_loss:.4f}")

torch.save(model.state_dict(), "fine_tuned_cross_age_model.pth")
print("Fine-tuned model saved.")

############## EVALUATION ##############
# adult test set

adult_test_dataset = BrainDataset(path_to_brats_test_imgs, path_to_brats_test_labels)
adult_test_loader = DataLoader(adult_test_dataset, batch_size=1, shuffle=False)

# pediatric test set
ped_test_dataset = BrainDataset(path_to_ped_test_imgs, path_to_ped_test_labels)
ped_test_loader = DataLoader(ped_test_dataset, batch_size=1, shuffle=False)

# evaluation
model.load_state_dict(torch.load("fine_tuned_cross_age_model.pth"))
model.to(device)

adult_results = evaluate_model(model, adult_test_loader, is_adult_label=1, device=device)
ped_results = evaluate_model(model, ped_test_loader, is_adult_label=0, device=device)

print("Adult Test:")
print(adult_results)

print("\nPediatric Test:")
print(ped_results)