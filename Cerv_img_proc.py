import PIL.Image as Image
import os as os
import os.path as os_path

train_img_input_dir = '/home/elijha/Documents/Data/Cervix/train'
train_img_output_dir = '/home/elijha/Documents/Data/Cervix/train_converted'
type_dir_list = os.listdir(train_img_input_dir)

for type_dir in type_dir_list:
    output_type_dir = os_path.join(train_img_output_dir,type_dir)
    if not os_path.isdir(output_type_dir):
        os.makedirs(output_type_dir)
    type_dir = os_path.join(train_img_input_dir, type_dir)
    if not os_path.isdir(type_dir):
        continue
    else:
        img_list = os.listdir(type_dir)
    for img_dir in img_list:
        # print()
        if not img_dir.split('.')[-1] == 'jpg':
            continue
        else:
            # 读出图片并resize成3000*2500
            img = Image.open(os_path.join(type_dir, img_dir)).resize([2500,3000])
            img.save(os_path.join(output_type_dir,img_dir))
            # img.save('test.bmp')
            # img = img.show()
