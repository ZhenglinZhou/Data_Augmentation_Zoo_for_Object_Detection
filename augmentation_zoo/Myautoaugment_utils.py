import numpy as np
import inspect
from PIL import Image, ImageOps
import cv2
import matplotlib.pyplot as plt
_MAX_LEVEL = 10.

def policy_custom():
    policy = []
    return policy

def policy_v0():
    policy = []
    return policy

def policy_v1():
    policy = []
    return policy

def policy_v2():
    policy = []
    return policy

def policy_v3():
    policy = []
    return policy

def policy_vtest():
    policy = [
        [('TranslateX_BBox', 1.0, 4)],
    ]
    return policy
def equalize(image):
    """
        均衡图像的直方图
    """
    if image.shape[0] == 0 or image.shape[1] == 0:
        return image

    if isinstance(image, np.ndarray):
        img = Image.fromarray(np.uint8(image))
    img = ImageOps.equalize(img)
    return np.array(img)

def translate_x(img, pixels, replace):
    """Equivalent of PIL Translate in X dimension."""
    if img.shape[0] == 0 or img.shape[1] == 0:
        return img

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)


    img = img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), fillcolor=replace)
    # img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, 0), fillcolor=replace)
    return np.array(img)


def translate_y(img, pixels, replace):
    """Equivalent of PIL Translate in Y dimension."""
    if img.shape[0] == 0 or img.shape[1] == 0:
        return img

    if isinstance(img, np.ndarray):
        img = Image.fromarray(np.uint8(img))

    img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), fillcolor=replace)
    return np.array(img)

def _clip_bbox(min_y, min_x, max_y, max_x):
    """Clip bounding box coordinates between 0 and 1.
    Args:
      min_y: Normalized bbox coordinate of type float between 0 and 1.
      min_x: Normalized bbox coordinate of type float between 0 and 1.
      max_y: Normalized bbox coordinate of type float between 0 and 1.
      max_x: Normalized bbox coordinate of type float between 0 and 1.
    Returns:
      Clipped coordinate values between 0 and 1.
    """
    min_y = np.clip(min_y, 0.0, 1.0)
    min_x = np.clip(min_x, 0.0, 1.0)
    max_y = np.clip(max_y, 0.0, 1.0)
    max_x = np.clip(max_x, 0.0, 1.0)
    return min_y, min_x, max_y, max_x


def _check_bbox_area(min_y, min_x, max_y, max_x, delta=0.05):
    """Adjusts bbox coordinates to make sure the area is > 0.
    Args:
      min_y: Normalized bbox coordinate of type float between 0 and 1.
      min_x: Normalized bbox coordinate of type float between 0 and 1.
      max_y: Normalized bbox coordinate of type float between 0 and 1.
      max_x: Normalized bbox coordinate of type float between 0 and 1.
      delta: Float, this is used to create a gap of size 2 * delta between
        bbox min/max coordinates that are the same on the boundary.
        This prevents the bbox from having an area of zero.
    Returns:
      Tuple of new bbox coordinates between 0 and 1 that will now have a
      guaranteed area > 0.
    """
    height = max_y - min_y
    width = max_x - min_x

    def _adjust_bbox_boundaries(min_coord, max_coord):
        # Make sure max is never 0 and min is never 1.
        max_coord = max(max_coord, 0.0 + delta)
        min_coord = max(min_coord, 1.0 - delta)
        return min_coord, max_coord

    if height == 0:
        min_y, max_y = _adjust_bbox_boundaries(min_y, max_y)

    if width == 0:
        min_x, max_x = _adjust_bbox_boundaries(min_x, max_x)

    return min_y, min_x, max_y, max_x

def _shift_bbox(bbox, image_height, image_width, pixels, shift_horizontal):
    """Shifts the bbox coordinates by pixels.
    Args:
      bbox: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
        of type float that represents the normalized coordinates between 0 and 1.
      image_height: Int, height of the image.
      image_width: Int, width of the image.
      pixels: An int. How many pixels to shift the bbox.
      shift_horizontal: Boolean. If true then shift in X dimension else shift in
        Y dimension.
    Returns:
      A tensor of the same shape as bbox, but now with the shifted coordinates.
    """
    pixels = pixels
    print(pixels)
    # Convert bbox to integer pixel locations.
    min_x = int(image_height * bbox[0])
    min_y = int(image_width * bbox[1])
    max_x = int(image_height * bbox[2])
    max_y = int(image_width * bbox[3])

    # if shift_horizontal:
    #     min_x += pixels
    #     max_x += pixels
    # else:
    #     min_y += pixels
    #     max_y += pixels


    if shift_horizontal:
        min_x = np.maximum(0, min_x - pixels)
        max_x = np.minimum(image_width, max_x - pixels)
    else:
        min_y = np.maximum(0, min_y - pixels)
        max_y = np.minimum(image_height, max_y - pixels)

    # Convert bbox back to floats.
    min_y = float(min_y) / float(image_height)
    min_x = float(min_x) / float(image_width)
    max_y = float(max_y) / float(image_height)
    max_x = float(max_x) / float(image_width)

    if max_x < 0. or min_x > 1.0 or max_y < 0. or min_y > 1.0:
        return None

    # Clip the bboxes to be sure the fall between [0, 1].
    min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x)
    min_y, min_x, max_y, max_x = _check_bbox_area(min_y, min_x, max_y, max_x)
    # return np.stack([min_y, min_x, max_y, max_x, bbox[4], bbox[5], bbox[6]])
    return np.stack([min_y, min_x, max_y, max_x])

def translate_bbox(image, bboxes, pixels, replace, shift_horizontal):
    """Equivalent of PIL Translate in X/Y dimension that shifts image and bbox.
        Args:
          image: 3D uint8 Tensor.
          bboxes: 2D Tensor that is a list of the bboxes in the image. Each bbox
            has 4 elements (min_y, min_x, max_y, max_x) of type float with values
            between [0, 1].
          pixels: An int. How many pixels to shift the image and bboxes
          replace: A one or three value 1D tensor to fill empty pixels.
          shift_horizontal: Boolean. If true then shift in X dimension else shift in
            Y dimension.
        Returns:
          A tuple containing a 3D uint8 Tensor that will be the result of translating
          image by pixels. The second element of the tuple is bboxes, where now
          the coordinates will be shifted to reflect the shifted image.
        """
    if shift_horizontal:
        image = translate_x(image, pixels, replace)
    else:
        image = translate_y(image, pixels, replace)

    # plt.imshow(image)
    # plt.show()

    # Convert bbox coordinates to pixel values.
    image_height = image.shape[0]
    image_width = image.shape[1]
    # pylint:disable=g-long-lambda

    wrapped_shift_bbox = lambda bbox: _shift_bbox(bbox, image_height, image_width, pixels, shift_horizontal)
    # 格式正确
    # pylint:enable=g-long-lambda
    bboxes = np.array([box for box in list(map(wrapped_shift_bbox, bboxes)) if box is not None])
    return image, bboxes

NAME_TO_FUNC = {
    'Equalize': equalize,
    'TranslateX_BBox': lambda image, bboxes, pixels, replace: translate_bbox(image, bboxes, pixels, replace,
                                                                          shift_horizontal=True),
    'TranslateY_BBox': lambda image, bboxes, pixels, replace: translate_bbox(image, bboxes, pixels, replace,
                                                                             shift_horizontal=False),
}
"""
NAME_TO_FUNC = {
    'AutoContrast': autocontrast,
    'Equalize': equalize,
    'Posterize': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'Cutout': cutout,
    'BBox_Cutout': bbox_cutout,
    'Rotate_BBox': rotate_with_bboxes,
    # pylint:disable=g-long-lambda
    'TranslateX_BBox': lambda image, bboxes, pixels, replace: translate_bbox(image, bboxes, pixels, replace,
                                                                             shift_horizontal=True),
    'TranslateY_BBox': lambda image, bboxes, pixels, replace: translate_bbox(image, bboxes, pixels, replace,
                                                                             shift_horizontal=False),
    'ShearX_BBox': lambda image, bboxes, level, replace: shear_with_bboxes(image, bboxes, level, replace,
                                                                           shear_horizontal=True),
    'ShearY_BBox': lambda image, bboxes, level, replace: shear_with_bboxes(image, bboxes, level, replace,
                                                                           shear_horizontal=False),
    # pylint:enable=g-long-lambda
    'Rotate_Only_BBoxes': rotate_only_bboxes,
    'ShearX_Only_BBoxes': shear_x_only_bboxes,
    'ShearY_Only_BBoxes': shear_y_only_bboxes,
    'TranslateX_Only_BBoxes': translate_x_only_bboxes,
    'TranslateY_Only_BBoxes': translate_y_only_bboxes,
    'Flip_Only_BBoxes': flip_only_bboxes,
    'Solarize_Only_BBoxes': solarize_only_bboxes,
    'Equalize_Only_BBoxes': equalize_only_bboxes,
    'Cutout_Only_BBoxes': cutout_only_bboxes,
}
"""


def _randomly_negate_tensor(tensor):
    """With 50% prob turn the tensor negative."""
    should_flip = np.floor(np.random.uniform() + 0.5).astype(np.bool_)
    if should_flip:
        final_tensor = tensor
    else:
        final_tensor = -tensor

    return final_tensor
def _rotate_level_to_arg(level):
    level = (level / _MAX_LEVEL) * 30.
    level = _randomly_negate_tensor(level)
    return (level,)


def _shrink_level_to_arg(level):
    """Converts level to ratio by which we shrink the image content."""
    if level == 0:
        return (1.0,)  # if level is zero, do not shrink the image
    # Maximum shrinking ratio is 2.9.
    level = 2. / (_MAX_LEVEL / level) + 0.9
    return (level,)


def _enhance_level_to_arg(level):
    return ((level / _MAX_LEVEL) * 1.8 + 0.1,)


def _shear_level_to_arg(level):
    level = (level / _MAX_LEVEL) * 0.3
    # Flip level to negative with 50% chance.
    level = _randomly_negate_tensor(level)
    return (level,)


def _translate_level_to_arg(level, translate_const):
    level = (level / _MAX_LEVEL) * float(translate_const)
    # Flip level to negative with 50% chance.
    level = _randomly_negate_tensor(level)
    return (level,)


def _bbox_cutout_level_to_arg(level, hparams):
    cutout_pad_fraction = (level / _MAX_LEVEL) * hparams['cutout_max_pad_fraction']
    return (cutout_pad_fraction,
            hparams['cutout_bbox_replace_with_mean'])

def level_to_arg(hparams):
    return {
        'AutoContrast': lambda level: (),
        'Equalize': lambda level: (),
        'Posterize': lambda level: (int((level / _MAX_LEVEL) * 4),),
        'Solarize': lambda level: (int((level / _MAX_LEVEL) * 256),),
        'SolarizeAdd': lambda level: (int((level / _MAX_LEVEL) * 110),),
        'Color': _enhance_level_to_arg,
        'Contrast': _enhance_level_to_arg,
        'Brightness': _enhance_level_to_arg,
        'Sharpness': _enhance_level_to_arg,
        'Cutout': lambda level: (int((level / _MAX_LEVEL) * hparams['cutout_const']),),
        # pylint:disable=g-long-lambda
        'BBox_Cutout': lambda level: _bbox_cutout_level_to_arg(
            level, hparams),
        'TranslateX_BBox': lambda level: _translate_level_to_arg(
            level, hparams['translate_const']),
        'TranslateY_BBox': lambda level: _translate_level_to_arg(
            level, hparams['translate_const']),
        # pylint:enable=g-long-lambda
        'ShearX_BBox': _shear_level_to_arg,
        'ShearY_BBox': _shear_level_to_arg,
        'Rotate_BBox': _rotate_level_to_arg,
        'Rotate_Only_BBoxes': _rotate_level_to_arg,
        'ShearX_Only_BBoxes': _shear_level_to_arg,
        'ShearY_Only_BBoxes': _shear_level_to_arg,
        # pylint:disable=g-long-lambda
        'TranslateX_Only_BBoxes': lambda level: _translate_level_to_arg(
            level, hparams['translate_bbox_const']),
        'TranslateY_Only_BBoxes': lambda level: _translate_level_to_arg(
            level, hparams['translate_bbox_const']),
        # pylint:enable=g-long-lambda
        'Flip_Only_BBoxes': lambda level: (),
        'Solarize_Only_BBoxes': lambda level: (int((level / _MAX_LEVEL) * 256),),
        'Equalize_Only_BBoxes': lambda level: (),
        # pylint:disable=g-long-lambda
        'Cutout_Only_BBoxes': lambda level: (
            int((level / _MAX_LEVEL) * hparams['cutout_bbox_const']),),
        # pylint:enable=g-long-lambda
    }

def bbox_wrapper(func):
    """Adds a bboxes function argument to func and returns unchanged bboxes."""

    def wrapper(images, bboxes, *args, **kwargs):
        return (func(images, *args, **kwargs), bboxes)

    return wrapper


def _parse_policy_info(name, prob, level, replace_value, augmentation_hparams):
    """Return the function that corresponds to `name` and update `level` param."""
    func = NAME_TO_FUNC[name]
    args = level_to_arg(augmentation_hparams)[name](level)

    # Check to see if prob is passed into function. This is used for operations
    # where we alter bboxes independently.
    # pytype:disable=wrong-arg-types

    if 'prob' in inspect.getfullargspec(func)[0]:
        # if 'prob' in inspect.signature(func)[0]:
        args = tuple([prob] + list(args))
    # pytype:enable=wrong-arg-types

    # Add in replace arg if it is required for the function that is being called.
    if 'replace' in inspect.getfullargspec(func)[0]:
        # Make sure replace is the final argument
        assert 'replace' == inspect.getfullargspec(func)[0][-1]
        args = tuple(list(args) + [replace_value])

    # Add bboxes as the second positional argument for the function if it does
    # not already exist.
    if 'bboxes' not in inspect.getfullargspec(func)[0]:
        func = bbox_wrapper(func)
    """
        args经过对应函数进行了修改，如：prob, replace
        func对参数存在没有bboxes进行了修改
        prob无修改
    """
    return (func, prob, args)

def _apply_func_with_prob(func, image, args, prob, bboxes):
    """Apply `func` to image w/ `args` as input with probability `prob`."""
    """
        调用函数
        assert 条件为false的时候执行
    """
    assert isinstance(args, tuple)
    assert 'bboxes' == inspect.getfullargspec(func)[0][1]

    # If prob is a function argument, then this randomness is being handled
    # inside the function, so make sure it is always called.
    """
        如果函数存在prob参数，则函数内部决定是否调用；
        如果函数没有prob参数，则在此处决定是否调用；
    """
    if 'prob' in inspect.getfullargspec(func)[0]:
        prob = 1.0

    # Apply the function with probability `prob`.
    should_apply_op = np.floor(np.random.uniform() + prob).astype(np.bool_)
    if should_apply_op:
        augmented_image, augmented_bboxes = func(image, bboxes, *args)
    else:
        augmented_image, augmented_bboxes = image, bboxes

    return augmented_image, augmented_bboxes

def select_and_apply_random_policy(policies, image, bboxes):
    """Select a random policy from `policies` and apply it to `image`."""
    policy_to_select = np.random.randint(low=0, high=len(policies))
    # Note that using tf.case instead of tf.conds would result in significantly
    # larger graphs and would even break export for some larger policies.
    for (i, policy) in enumerate(policies):
        if policy_to_select == i:
            image, bboxes = policy(image, bboxes)
        else:
            image, bboxes = image, bboxes
    return (image, bboxes)

def build_and_apply_nas_policy(policies, image, bboxes, augmentation_hparams):
    # replace_value = (128, 128, 128)
    replace_value = (0, 0, 0)
    tf_policies = []
    for policy in policies:
        tf_policy = []
        """
            policies是对应的v0-3
            policy是sub_policy
            policy_info是具体的规则
            tf_policy是对应的sub_policy
        """
        for policy_info in policy:
            """
                policy_info = [name, prob, level]
            """
            policy_info = list(policy_info) + [replace_value, augmentation_hparams]
            tf_policy.append(_parse_policy_info(*policy_info)) #list中的每一项作为一个参数

        def make_final_policy(tf_policy_):
            def final_policy(image_, bboxes_):
                for func, prob, args in tf_policy_:
                    image_, bboxes_ = _apply_func_with_prob(func, image_, args, prob, bboxes_)
                return image_, bboxes_
            return final_policy
        samples = make_final_policy(tf_policy)
        tf_policies.append(samples)
    augmented_images, augmented_bboxes = select_and_apply_random_policy(tf_policies, image, bboxes)
    # If no bounding boxes were specified, then just return the images.
    return (augmented_images, augmented_bboxes)

def distort_image_with_autoaugment(image, bboxes, augmentation_name):
    available_policies = {'v0': policy_v0, 'v1': policy_v1, 'v2': policy_v2,
                          'v3': policy_v3, 'test': policy_vtest,
                          'custom': policy_custom}
    if augmentation_name not in available_policies:
        raise ValueError('Invalid augmentation_name: {}'.format(augmentation_name))
    """
        选择一个策略v0-3，算法会等概率的选择其中的一个sub_policy执行。
    """
    policy = available_policies[augmentation_name]()

    augmentation_hparams = {
        "cutout_max_pad_fraction": 0.75,
        "cutout_bbox_replace_with_mean": False,
        "cutout_const": 100,
        "translate_const": 250,
        "cutout_bbox_const": 50,
        "translate_bbox_const": 120}

    return build_and_apply_nas_policy(policy, image, bboxes, augmentation_hparams)

