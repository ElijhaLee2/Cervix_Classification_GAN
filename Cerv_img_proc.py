import PIL.Image as Image
import os as os
import os.path as os_path

train_img_root = '/media/elijha/Software/data/train'
type_dir_list = os.listdir(train_img_root)

for type_dir in type_dir_list:
    type_dir = os_path.join(train_img_root, type_dir)
    if not os_path.isdir(type_dir):
        continue
    else:
        img_list = os.listdir(type_dir)
    for img_dir in img_list:
        # print()
        if not img_dir.split('.')[-1] == 'jpg':
            continue
        else:
            img = Image.open(os_path.join(type_dir, img_dir))
            print()
