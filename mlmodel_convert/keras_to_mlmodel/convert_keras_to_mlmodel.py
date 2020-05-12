"""
Reads keras weights and creates .mlmodel model with coremltools.
For each conversion method, you can use `class_labels = output_labels` or not.
To see the differences in detail, use the Netron application (https://github.com/lutzroeder/netron).

"""

from keras.models import load_model
import coremltools
import argparse
import sys


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', help='Image size for YoloV3 model.',
                        default=416)
    parser.add_argument('--h5_path', help='Path to Darknet weights file.',
                        default='../../pretrain_models/yolov2/yolov2.h5')
    parser.add_argument('--mlmodel_path', help='Path to output ML-Model file.',
                        default='../../pretrain_models/yolov2/yolov2_keras.mlmodel')
    return parser.parse_args(argv)


args = parse_arguments(sys.argv[1:])


def convert_h5_to_mlmodel():
    # YOLO V3
    # coreml_model = coremltools.converters.keras.convert(args.h5_path, input_names=['image'], output_names=[
    #     'output1', 'output2', 'output3'], image_scale=1 / 255., image_input_names='image', input_name_shape_dict={'image': [None, args.img_size, args.img_size, 3]})

    # YOLO V3-Tiny
    # coreml_model = coremltools.converters.keras.convert(args.h5_path, input_names=['image'], output_names=[
    #     'output1', 'output2'], image_scale=1 / 255., image_input_names='image', input_name_shape_dict={'image': [None,  args.img_size,  args.img_size, 3]})

    # YOLO V2
    coreml_model = coremltools.converters.keras.convert(args.h5_path, input_names=['image'], output_names=[
        'grid'], image_scale=1/255., image_input_names='image', input_name_shape_dict={'image': [None,  args.img_size,  args.img_size, 3]})

    # coreml_model.input_description['image'] = 'Input image'
    # coreml_model.output_description['grid'] = 'The 13x13 grid'

    coreml_model.author = 'Duc Phan'
    coreml_model.license = 'Public Domain'
    coreml_model.short_description = "The YOLOv3 network from the paper 'YOLOv3: An Incremental Improvement'"

    coreml_model.save(args.mlmodel_path)


if __name__ == "__main__":
    convert_h5_to_mlmodel()