import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import time

from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weigths = torch.load('best.pt')
model = weigths['model']
model = model.half().to(device)
_ = model.eval()

img_path = '1.jpg'

img = cv2.imread(img_path)

# Get the frame width and height.
h,w,c = img.shape
frame_width = w
frame_height = h


orig_image = img
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
image = letterbox(image, (frame_width), stride=64, auto=True)[0]
image_ = image.copy()
image = transforms.ToTensor()(image)
image = torch.tensor(np.array([image.numpy()]))
image = image.to(device)
image = image.half()

with torch.no_grad():
    output, _ = model(image)

output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], kpt_label=True)
output = output_to_keypoint(output)
nimg = image[0].permute(1, 2, 0) * 255
nimg = nimg.cpu().numpy().astype(np.uint8)
nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
for idx in range(output.shape[0]):
    # plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

    # Comment/Uncomment the following lines to show bounding boxes around persons.
    xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
    xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)

    plate_roi = nimg[int(ymin):int(ymax),int(xmin):int(xmax)]
    cv2.imshow('Plate',plate_roi)

    cv2.putText(nimg, "Number Plate", (int(xmin), int(ymin)-5), cv2.FONT_HERSHEY_SIMPLEX,1, (228, 79, 215), 2)
    cv2.rectangle(
        nimg,
        (int(xmin), int(ymin)),
        (int(xmax), int(ymax)),
        color=(228, 79, 215),
        thickness=1,
        lineType=cv2.LINE_AA
    )

# Convert from BGR to RGB color format.
cv2.imwrite('result.jpg',nimg)