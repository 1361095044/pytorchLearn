from PIL import Image
from torchvision import transforms


# 用法
# 通过transforms.ToTensor去看两个问题

# tensor数据类型

img_path = 'train/ants_image/0013035.jpg'
img = Image.open(img_path)
# 如何使用transforms
tensor_1 = transforms.ToTensor()
tensor_img = tensor_1(img)  # 返回tensor类型的图片
print(tensor_img)