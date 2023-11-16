import math

import torch
from torch.nn import CrossEntropyLoss
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50, deeplabv3_resnet101
from torchvision.ops import sigmoid_focal_loss
from tqdm import tqdm
import numpy as np
from os import listdir
from PIL import Image
from utils import train_mapping_to_rgb, load_images, load_labels, invert_channels, augment_pairs, to_tensor, imgroot, val_imgroot, project_out
from PIL import ImageFilter

losses = []
accuracies = []
val_losses = []
val_accuracies = []


def train():
    images = listdir(imgroot)
    val_images = listdir(val_imgroot)

    batch_size = 4
    epochs = 101
    lr = 0.001
    lr_decay = 0.97

    # Resnet50 training: ~3min/epoch
    # Resnet101 training: ~3.5min/epoch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #model = deeplabv3_resnet101(num_classes=19)
    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, 19, kernel_size=(1,1), stride=(1,1))
    model.to(device)
    model.eval()
    # model.load_state_dict(torch.load("models/model.pth"))

    criterion = CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        #for g in optimizer.param_groups:
        #    g['lr'] = lr * lr_decay**epoch

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
            bar.set_description(f"Epoch {epoch} loss: {acum_loss/(i*batch_size+1)} accuracy: {round((correct/(total+1))*100, 1)}% val loss:{val_loss/(val_ct+1)} val accuracy: {round((val_correct / (val_total + 1)) * 100, 1)}")

            image_names = images[i:i+batch_size]
            
            #batch_img = load_images(image_names, filter=ImageFilter.GaussianBlur(3))
            batch_img = load_images(image_names)
            batch_labels = load_labels(image_names)
            #batch_img, batch_labels = augment_pairs(batch_img, batch_labels)

            batch_img = to_tensor(invert_channels(batch_img), device=device)
            batch_labels = to_tensor(batch_labels, dtype=torch.long, device=device)

            out = model(batch_img)["out"]
            loss = criterion(out, batch_labels)
            loss.backward()
            optimizer.step()
            acum_loss += loss.item()

            out = out.cpu().detach().numpy()
            if (i*batch_size) % 200 == 0:
                # Calculate loss and accuracy of train
                projected = project_out(out)
                batch_labels = batch_labels.cpu().detach().numpy()
                correct += (projected == batch_labels).sum().item()
                total += batch_labels.size

                #Image.fromarray(projected[0]).show()

                # Calculate loss and accuracy of validation
                np.random.shuffle(val_images)
                #val_imgs = torch.tensor(load_images(val_images[:batch_size], True), dtype=torch.float32).to(device)
                #val_labels = torch.tensor(load_labels(val_images[:batch_size], True), dtype=torch.long).to(device)
                val_imgs = to_tensor(invert_channels(load_images(val_images[:batch_size], True)), device=device)
                val_labels = to_tensor(load_labels(val_images[:batch_size], True), dtype=torch.long, device=device)

                out = model(val_imgs)['out']
                val_loss += criterion(out, val_labels).item()
                out = out.cpu().detach().numpy()
                projected = project_out(out)
                batch_labels = val_labels.cpu().detach().numpy()
                val_correct += (projected == batch_labels).sum().item()
                val_total += batch_labels.size
                val_ct += batch_size


        accuracies.append(round((correct/total)*100, 1))
        losses.append(acum_loss/(i*batch_size))
        val_accuracies.append(round((val_correct/val_total)*100, 1))
        val_losses.append(val_loss/val_ct)

        # Save metrics and model after each epoch
        torch.save(model.state_dict(), f"models/model_epoch_{epoch}.pth")
        np.save("accuracies.npy", np.array(accuracies))
        np.save("losses.npy", np.array(losses))
        np.save("val_accuracies.npy", np.array(val_accuracies))
        np.save("val_losses.npy", np.array(val_losses))

    if epoch % 5 == 0:
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