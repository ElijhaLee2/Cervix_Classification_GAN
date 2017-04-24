import PIL.Image as Image
import os as os
import os.path as os_path

train_img_input_dir = '/home/elijha/Documents/Data/Cervix/train'
train_img_output_dir = '/home/elijha/Documents/Data/Cervix/train_converted'
type_dir_list = os.listdir(train_img_input_dir)

for type_dir in type_dir_list:
    # type_dir = 'Type_3'
    print(type_dir)
    output_type_dir = os_path.join(train_img_output_dir, type_dir)
    if not os_path.isdir(output_type_dir):
        os.makedirs(output_type_dir)
    type_dir = os_path.join(train_img_input_dir, type_dir)
    if not os_path.isdir(type_dir):
        continue
    else:
        img_list = os.listdir(type_dir)

        total = len(img_list)
        i = 1.0
        for img_dir in img_list:
            # print()
            if not img_dir.split('.')[-1] == 'jpg':
                continue
            else:
                # 448*448
                print('(%.2f%%) Processing: ' % (i / total * 100) + img_dir, end='......')
                try:
                    img = Image.open(os_path.join(type_dir, img_dir)).resize([448, 448])
                    img.save(os_path.join(output_type_dir, img_dir))
                    i += 1
                    print('Finished!')
                except Exception as e:
                    print(img_dir)
                    print(e)