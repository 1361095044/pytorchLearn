from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter('logs')
# 图片处理
img_path = 'train/ants_image/0013035.jpg'
img = Image.open(img_path)

# ToTensor
trans_toTensor = transforms.ToTensor()
tensor_img = trans_toTensor(img)
writer.add_image('Tensor_img', tensor_img)

# Normalize
trans_norm = transforms.Normalize([1, 1, 1], [1, 1, 2])
norm_img = trans_norm(tensor_img)
writer.add_image('Normalize_img', norm_img, 1)  # 1指的step

# Resize
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_toTensor(img_resize)
writer.add_image('Resize_img', img_resize, 0)

# Compose
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_toTensor])
img_resize_2 = trans_compose(img)
writer.add_image('Resize_img', img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, trans_toTensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image('RandomCrop', img_crop, i)



writer.close()
