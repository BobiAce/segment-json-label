# coding: utf-8
import argparse
import json
import os
import os.path as osp
import warnings

import PIL.Image
import yaml

from labelme import utils
import base64
from skimage import img_as_ubyte
import cv2
import numpy as np

###=======================================================
###=============为得到的label.png进行着色====================
###=======================================================
# color_map = labelcolormap(2)
# print (color_map)
# img = PIL.Image.open('/home/jinbo/data_all/traindata/labeltrain/1189459186_json/label.png')
# img = np.array(img)
# dst = color.label2rgb(img, colors=color_map[1:], bg_label=0, bg_color=(0, 0, 0))
# io.imsave('/home/jinbo/data_all/traindata/labeltrain/1189459186_json/xx.png', dst)

# Get the specified bit value
def bitget(byteval, idx):
  return ((byteval & (1 << idx)) != 0)

# Create label-color map, label --- [R G B]
#  0 --- [  0   0   0],  1 --- [128   0   0],  2 --- [  0 128   0]
#  3 --- [128 128   0],  4 --- [  0   0 128],  5 --- [128   0 128]
#  6 --- [  0 128 128],  7 --- [128 128 128],  8 --- [ 64   0   0]
#  9 --- [192   0   0], 10 --- [ 64 128   0], 11 --- [192 128   0]
# 12 --- [ 64   0 128], 13 --- [192   0 128], 14 --- [ 64 128 128]
# 15 --- [192 128 128], 16 --- [  0  64   0], 17 --- [128  64   0]
# 18 --- [  0 192   0], 19 --- [128 192   0], 20 --- [  0  64 128]
def labelcolormap(N=256):
  color_map = np.zeros((N, 3))
  for n in xrange(N):
    id_num = n
    r, g, b = 0, 0, 0
    for pos in xrange(8):
      r = np.bitwise_or(r, (bitget(id_num, 0) << (7-pos)))
      g = np.bitwise_or(g, (bitget(id_num, 1) << (7-pos)))
      b = np.bitwise_or(b, (bitget(id_num, 2) << (7-pos)))
      id_num = (id_num >> 3)
    color_map[n, 0] = r
    color_map[n, 1] = g
    color_map[n, 2] = b
  return color_map/255

'''
批量把labelme标注的json文件转换为16位的label.png 和 8位的mask.png图像
'''
def main():
    warnings.warn("This script is aimed to demonstrate how to convert the\n"
                  "JSON file to a single image dataset, and not to handle\n"
                  "multiple JSON files to generate a real-use dataset.")
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', default='jsonFile/', help='path to load json_file. . ')
    parser.add_argument('-o', '--out', default='labeltrain_test',help='path to output mask. . ')
    # parser.add_argument('json_file')
    # parser.add_argument('-o', '--out', default=None)
    args = parser.parse_args()

    json_file = args.json_file
    if args.out is None:
        out_dir = osp.basename(json_file).replace('.', '_')
        out_dir = osp.join(osp.dirname(json_file), out_dir)
    else:
        out_dir = args.out
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    # print("jsonfile:",json_file)
    # print(out_dir)

    count = os.listdir(json_file)
    # print(count[0:10])
    #
    for i in range(0, len(count)):
        path = os.path.join(json_file, count[i])
        if os.path.isfile(path):
            data = json.load(open(path))

            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')
            img = utils.img_b64_to_arr(imageData)

            lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])
            captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]

            lbl_viz = utils.draw_label(lbl, img, captions)
            ####====================================================================
            ####=======creat dir save label ========================================
            json_out_dir = osp.basename(path).replace('.', '_')
            save_file_name = osp.join(osp.dirname(__file__),out_dir ,json_out_dir)
            # print(save_file_name)
            # print(abc)
            #creat dir save label 
            if not osp.exists(save_file_name):
                os.mkdir(save_file_name)
            # PIL.Image.fromarray(img).save(out_dir1 + '\\' + save_file_name + '_img.png')
            PIL.Image.fromarray(lbl).save(save_file_name + '/' + 'label.png')
            PIL.Image.fromarray(lbl_viz).save(save_file_name + '/' + 'label_viz.png')
            ####=====================================================================

            #========================================================================
            #==============transform label.png to uint8 for train====================
            mask_path = out_dir
            mask_name = osp.basename(path).split(".")[0]
            mask_dst = img_as_ubyte(lbl)  # mask_pic
            print('pic2_deep:', mask_dst.dtype)
            cv2.imwrite(mask_path + '/' + mask_name + '.png', mask_dst)
            ####======================================================================



            with open(osp.join(save_file_name, 'label_names.txt'), 'w') as f:
                for lbl_name in lbl_names:
                    f.write(lbl_name + '\n')

            warnings.warn('info.yaml is being replaced by label_names.txt')
            info = dict(label_names=lbl_names)
            with open(osp.join(save_file_name, 'info.yaml'), 'w') as f:
                yaml.safe_dump(info, f, default_flow_style=False)

            print('Saved to: %s' %save_file_name)


if __name__ == '__main__':
    main()










'''
以上代码可以将json文件中的label存储为png图像文件。但是存在一个问题：对于多类分割任务
任意一张图可能不包含所有分类。因此整个文件夹下生成的所有label图像中，
不同图像中的相同类别的目标在label.png中可能对应不同的灰度值，使标注的label不具备统一性，因而出错。

通过建立全局标签值字典，可以控制label图像中目标对应相同的灰度值，从而保证标签在所有图像中的一致性。
同时我们修改了图像存储时的命名，使其命名与原图命名对应。
'''
# # coding: utf-8
# import argparse
# import json
# import os
# import os.path as osp
# import warnings
# import copy
#
# import numpy as np
# import PIL.Image
# from skimage import io
# import yaml
#
# from labelme import utils
#
# NAME_LABEL_MAP = {
#     '_background_': 0,
#     "baseball_diamond": 1,
#     "tennis_court": 2,
#     "basketball_court": 3,
#     "ground_track_field": 4,
# }
#
# LABEL_NAME_MAP = {
#     0: '_background_',
#     1: "airplane",
#     2: "ship",
#     3: "storage_tank",
#     4: "baseball_diamond",
#     5: "tennis_court",
#     6: "basketball_court",
#     7: "ground_track_field",
#     8: "harbor",
#     9: "bridge",
#     10: "vehicle",
# }
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('json_file')
#     parser.add_argument('-o', '--out', default=None)
#     args = parser.parse_args()
#
#     json_file = args.json_file
#
#     list = os.listdir(json_file)
#     for i in range(0, len(list)):
#         path = os.path.join(json_file, list[i])
#         filename = list[i][:-5]       # .json
#         if os.path.isfile(path):
#             data = json.load(open(path))
#             img = utils.image.img_b64_to_arr(data['imageData'])
#             lbl, lbl_names = utils.shape.labelme_shapes_to_label(img.shape, data['shapes'])  # labelme_shapes_to_label
#
#             # modify labels according to NAME_LABEL_MAP
#             lbl_tmp = copy.copy(lbl)
#             for key_name in lbl_names:
#                 old_lbl_val = lbl_names[key_name]
#                 new_lbl_val = NAME_LABEL_MAP[key_name]
#                 lbl_tmp[lbl == old_lbl_val] = new_lbl_val
#             lbl_names_tmp = {}
#             for key_name in lbl_names:
#                 lbl_names_tmp[key_name] = NAME_LABEL_MAP[key_name]
#
#             # Assign the new label to lbl and lbl_names dict
#             lbl = np.array(lbl_tmp, dtype=np.int8)
#             lbl_names = lbl_names_tmp
#
#             captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
#             lbl_viz = utils.draw.draw_label(lbl, img, captions)
#             out_dir = osp.basename(list[i]).replace('.', '_')
#             out_dir = osp.join(osp.dirname(list[i]), out_dir)
#             if not osp.exists(out_dir):
#                 os.mkdir(out_dir)
#
#             PIL.Image.fromarray(img).save(osp.join(out_dir, '{}.png'.format(filename)))
#             PIL.Image.fromarray(lbl).save(osp.join(out_dir, '{}_gt.png'.format(filename)))
#             PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, '{}_viz.png'.format(filename)))
#
#             with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
#                 for lbl_name in lbl_names:
#                     f.write(lbl_name + '\n')
#
#             warnings.warn('info.yaml is being replaced by label_names.txt')
#             info = dict(label_names=lbl_names)
#             with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
#                 yaml.safe_dump(info, f, default_flow_style=False)
#
#             print('Saved to: %s' % out_dir)
#
#
# if __name__ == '__main__':
#     main()