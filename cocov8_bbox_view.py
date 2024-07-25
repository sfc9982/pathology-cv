import json
import os

import cv2

train_json = '/mnt/sdc1/Cervical_Datasets/we_data/coco/annotations/val.json'
train_path = '/mnt/sdc1/Cervical_Datasets/we_data/coco/val'


def visualization_bbox(num_image, json_path, img_path):
    with open(json_path) as annotations:
        annotation_json = json.load(annotations)

    print('the annotation_json num_key is:', len(annotation_json))
    print('the annotation_json key is:', annotation_json.keys())
    print('the annotation_json num_images is:', len(annotation_json['images']))

    # image_name = annotation_json['images'][num_image - 1]['file_name']  # 读取图片名
    # id = annotation_json['images'][num_image - 1]['id']  # 读取图片id

    image_name = ''
    id = 0
    for i in range(len(annotation_json['images'][::])):
        img_id = int(annotation_json['images'][i - 1]['id'])
        if img_id == num_image:
            image_name = annotation_json['images'][i - 1]['file_name']
            id = annotation_json['images'][i - 1]['id']

    image_path = os.path.join(img_path, str(image_name).zfill(5))
    image = cv2.imread(image_path, 1)
    num_bbox = 0

    text_width = 12
    text_height = 15
    font_scale = 0.5
    classes = ['ASC-US', 'ASC-H', 'LSIL', 'HSIL']
    for i in range(len(annotation_json['annotations'][::])):
        if annotation_json['annotations'][i - 1]['image_id'] == id:
            num_bbox = num_bbox + 1
            x, y, w, h = annotation_json['annotations'][i - 1]['bbox']
            image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 127, 127), 2)
            text = classes[int(annotation_json['annotations'][i - 1]['category_id']) - 1]
            width = len(text) * text_width
            cv2.rectangle(image, (x + 2, y + 2), (x + width + 2, y + text_height + 2), (0, 0, 0), -1)
            cv2.putText(image, text, (x + 4, y + text_height), cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 255, 255))

    print('The unm_bbox of the display image is:', num_bbox)

    cv2.imwrite(str(id) + "_val.jpg", image)


if __name__ == "__main__":
    visualization_bbox(213, train_json, train_path)
