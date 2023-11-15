import math

import torch
from torch.nn import CrossEntropyLoss
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50
from torchvision.ops import sigmoid_focal_loss
from tqdm import tqdm
import numpy as np
from os import listdir
from PIL import Image
from utils import train_mapping_to_rgb

val_imgroot = './data2/val/img'
val_labelroot = './data2/val/labels'
imgroot = './data2/train/img'
labelroot = './data2/train/labels'

losses = []
accuracies = []
val_losses = []
val_accuracies = []

def project_out(out, colored=False):
    projected = np.zeros(shape=(4, 256, 512))
    for i in range(len(out)):
        for w in range(out.shape[2]):
            for h in range(out.shape[3]):
                max = -999
                _ = -1
                for c in range(out.shape[1]):
                    if out[i][c][w][h] > max:
                        max = out[i][c][w][h]
                        _ = c
                if colored:
                    _ = train_mapping_to_rgb[_]
                projected[i][w][h] = _
    return projected

def load_images(image_names, validation=False, device="cuda"):
    return np.array([np.moveaxis(np.array(Image.open(f"{imgroot if not validation else val_imgroot}/{x}")), -1, 0) for x in image_names])

def load_labels(image_names, validation=False):
    return np.array([np.array(Image.open(f"{labelroot if not validation else val_labelroot}/{x}")) for x in image_names])

def train():
    images = listdir(imgroot)
    val_images = listdir(val_imgroot)

    batch_size = 4
    epochs = 50

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = deeplabv3_resnet50(num_classes=19)
    model.to(device)
    model.eval()
    # model.load_state_dict(torch.load("models/model.pth"))

    criterion = CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        np.random.shuffle(images)
        acum_loss = 0
        correct = 0
        total = 0

        val_loss = 0
        val_correct = 0
        val_total = 0
        val_ct = 0
        bar = tqdm(range(math.floor(len(images)/batch_size)))
        for i in bar:
            optimizer.zero_grad()
            bar.set_description(f"Epoch {epoch} loss: {acum_loss/(i*4+1)} accuracy: {round((correct/(total+1))*100, 1)}% val loss:{val_loss/(val_ct+1)} val accuracy: {round((val_correct / (val_total + 1)) * 100, 1)}")

            image_names = images[i:i+batch_size]
            batch_img = torch.tensor(load_images(image_names), dtype=torch.float32).to(device)
            batch_labels = torch.tensor(load_labels(image_names), dtype=torch.long).to(device)
            out = model(batch_img)["out"]
            loss = criterion(out, batch_labels)
            loss.backward()
            optimizer.step()
            acum_loss += loss.item()

            out = out.cpu().detach().numpy()
            if i % 50 == 0:
                # Calculate loss and accuracy of train
                projected = project_out(out)
                batch_labels = batch_labels.cpu().detach().numpy()
                correct += (projected == batch_labels).sum().item()
                total += batch_labels.size

                Image.fromarray(projected[0]).show()

                # Calculate loss and accuracy of validation
                np.random.shuffle(val_images)
                val_imgs = torch.tensor(load_images(val_images, True), dtype=torch.float32).to(device)
                val_labels = torch.tensor(load_labels(val_images, True), dtype=torch.long).to(device)

                out = model(val_imgs)['out']
                val_loss += criterion(out, val_labels).item()
                out = out.cpu().detach().numpy()
                projected = project_out(out)
                batch_labels = val_labels.cpu().detach().numpy()
                val_correct += (projected == batch_labels).sum().item()
                val_total += batch_labels.size
                val_ct += 4


        accuracies.append(round((correct/total)*100, 1))
        losses.append(acum_loss/(i*4))
        val_accuracies.append(round((val_correct/val_total)*100, 1))
        val_losses.append(val_loss/val_ct)

        # Save metrics and model after each epoch
        torch.save(model.state_dict(), f"models/model_epoch_{epoch}.pth")
        np.save("accuracies.npy", np.array(accuracies))
        np.save("losses.npy", np.array(losses))
        np.save("val_accuracies.npy", np.array(val_accuracies))
        np.save("val_losses.npy", np.array(val_losses))

    torch.save(model.state_dict(), "models/model.pth")

if __name__ == "__main__":
    train()

# Load up in batches
# batch = list(zip(images[i:i+batch_size], labels[i:i+batch_size]))
# batch = (images[i:i+batch_size], labels[i:i+batch_size])
# Load tensors
# for j in range(batch_size):
#    if batch[0][j] != batch[1][j]:
#        print("Mismatching image and labels")
#        exit(1)
#    img = torch.tensor(np.moveaxis(np.array(Image.open(f"{imgroot}/{batch[0][j]}")), -1, 0), dtype=torch.float32).to(device)
#    label = torch.tensor(np.array(Image.open(f"{labelroot}/{batch[0][j]}")), dtype=torch.long).to(device)
#    batch[0][j] = img
#    batch[1][j] = label

# for img, label in batch:
#    out = model(img.unsqueeze(0))["out"]
#    out.shape
#    label.shape
#    loss = criterion(out, label.unsqueeze(0))
#    loss.backward()
# acum_loss += loss.item()

#    correct += (out == label).sum().item()
#    total += label.numel()