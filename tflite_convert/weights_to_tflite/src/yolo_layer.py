import tensorflow as tf


class YOLOLayer(tf.keras.layers.Layer):

    def __init__(self, num_classes, anchors, input_dims, **kwargs):
        self.num_classes = num_classes
        self.anchors = anchors
        self.input_dims = input_dims

        super(YOLOLayer, self).__init__(**kwargs)

    def call(self, prediction, **kwargs):
        batch_size = tf.shape(prediction)[0]
        stride = self.input_dims[0] // tf.shape(prediction)[1]
        grid_size = self.input_dims[0] // stride
        num_anchors = len(self.anchors)

        prediction = tf.reshape(prediction,
                                shape=(batch_size, num_anchors * grid_size * grid_size, self.num_classes + 5))

        box_xy = tf.sigmoid(prediction[:, :, :2])  # t_x (box x and y coordinates)
        objectness = tf.sigmoid(prediction[:, :, 4])  # p_o (objectness score)
        objectness = tf.expand_dims(objectness, 2)  # To make the same number of values for axis 0 and 1

        grid = tf.range(grid_size)
        a, b = tf.meshgrid(grid, grid)

        x_offset = tf.reshape(a, (-1, 1))
        y_offset = tf.reshape(b, (-1, 1))

        x_y_offset = tf.concat((x_offset, y_offset), axis=1)
        x_y_offset = tf.tile(x_y_offset, (1, num_anchors))
        x_y_offset = tf.reshape(x_y_offset, (-1, 2))
        x_y_offset = tf.expand_dims(x_y_offset, 0)
        x_y_offset = tf.cast(x_y_offset, dtype='float32')

        box_xy += x_y_offset

        # Log space transform of the height and width
        anchors = tf.cast([(a[0] / stride, a[1] / stride) for a in self.anchors], dtype='float32')
        anchors = tf.tile(anchors, (grid_size * grid_size, 1))
        anchors = tf.expand_dims(anchors, 0)

        box_wh = tf.exp(prediction[:, :, 2:4]) * anchors

        # Sigmoid class scores
        class_scores = tf.sigmoid(prediction[:, :, 5:])

        # Resize detection map back to the input image size
        stride = tf.cast(stride, dtype='float32')
        box_xy *= stride
        box_wh *= stride

        # Convert centoids to top left coordinates
        box_xy -= box_wh / 2

        return tf.keras.layers.Concatenate(axis=2)([box_xy, box_wh, objectness, class_scores])

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        num_anchors = len(self.anchors)
        stride = self.input_dims[0] // input_shape[1]
        grid_size = self.input_dims[0] // stride
        num_bboxes = num_anchors * grid_size * grid_size

        shape = (batch_size, num_bboxes, self.num_classes + 5)

        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(YOLOLayer, self).get_config()
        config = {
            'num_classes': self.num_classes,
            'anchors': self.anchors,
            'input_dims': self.input_dims
        }

        return dict(list(base_config.items()) + list(config.items()))

