import numpy as np
from dataclasses import dataclass
from .blazeface_constants import AnchorOptions


@dataclass
class DetectionResults:
    boxes: np.array
    keypoints: np.array
    scores: float


@dataclass
class Anchor:
    x_centre: float
    y_centre: float
    height: int
    width: int


def calculate_scale(min_scale, max_scale, stride_index, num_strides):
    return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1.0)


def generate_anchors() -> np.ndarray:
    """
    Literal translaation of
    https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/ssd_anchors_calculator.cc
    """
    options = AnchorOptions()
    strides_size = len(options.STRIDES)
    assert options.NUM_LAYERS == strides_size

    anchors = []
    layer_id = 0
    while layer_id < strides_size:
        anchor_height = []
        anchor_width = []
        aspect_ratios = []
        scales = []

        # For same strides, we merge the anchors in the same order.
        last_same_stride_layer = layer_id
        while (last_same_stride_layer < strides_size) and \
                (options.STRIDES[last_same_stride_layer] == options.STRIDES[layer_id]):
            scale = calculate_scale(options.MIN_SCALE,
                                    options.MAX_SCALE,
                                    last_same_stride_layer,
                                    strides_size)

            if last_same_stride_layer == 0 and options.REDUCE_BOXES_IN_LOWEST_LAYER:
                # UNUSED
                # For first layer, it can be specified to use predefined anchors.
                aspect_ratios.append(1.0)
                aspect_ratios.append(2.0)
                aspect_ratios.append(0.5)
                scales.append(0.1)
                scales.append(scale)
                scales.append(scale)
            else:
                for aspect_ratio in options.ASPECT_RATIOS:
                    aspect_ratios.append(aspect_ratio)
                    scales.append(scale)

                if options.INTERPOLATED_SCALE_ASPECT_RATIO > 0.0:
                    scale_next = 1.0 if last_same_stride_layer == strides_size - 1 \
                        else calculate_scale(options.MIN_SCALE,
                                             options.MAX_SCALE,
                                             last_same_stride_layer + 1,
                                             strides_size)
                    scales.append(np.sqrt(scale * scale_next))
                    aspect_ratios.append(options.INTERPOLATED_SCALE_ASPECT_RATIO)

            last_same_stride_layer += 1

        for i in range(len(aspect_ratios)):
            ratio_sqrts = np.sqrt(aspect_ratios[i])
            anchor_height.append(scales[i] / ratio_sqrts)
            anchor_width.append(scales[i] * ratio_sqrts)

        stride = options.STRIDES[layer_id]
        feature_map_height = int(np.ceil(options.INPUT_SIZE_HEIGHT / stride))
        feature_map_width = int(np.ceil(options.INPUT_SIZE_WIDTH / stride))

        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for anchor_id in range(len(anchor_height)):
                    x_center = (x + options.ANCHOR_OFFSET_X) / feature_map_width
                    y_center = (y + options.ANCHOR_OFFSET_Y) / feature_map_height

                    new_anchor = [x_center, y_center, 0, 0]
                    if options.FIXED_ANCHOR_SIZE:
                        new_anchor[2] = 1.0
                        new_anchor[3] = 1.0
                    else:
                        # UNUSED
                        new_anchor[2] = anchor_width[anchor_id]
                        new_anchor[3] = anchor_height[anchor_id]
                    anchors.append(new_anchor)

        layer_id = last_same_stride_layer

    assert len(anchors) == 896
    # convert anchors from list into numpy array
    anchors = np.array(anchors, dtype=np.float32)

    return anchors
