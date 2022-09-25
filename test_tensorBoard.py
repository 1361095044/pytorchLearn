import numpy as np
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

writer = SummaryWriter('logs')
image_path = 'train/ants_image/0013035.jpg'
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)
writer.add_image('test', img_array, 1, dataformats='HWC')
for i in range(100):
    writer.add_scalar(tag='y=x', scalar_value=i, global_step=i)

writer.close()
