# Data_Augmentation_Zoo_for_Object_Detection
## Background

This project is built for testing multiple data augmentations for object detection:
1. Zoph B, Cubuk E D, Ghiasi G, et al. Learning data augmentation strategies for object detection[J]. arXiv preprint arXiv:1906.11172, 2019.
[pdf](https://arxiv.org/pdf/1906.11172.pdf "pdf") | [github](https://github.com/tensorflow/tpu/blob/master/models/official/detection/utils/autoaugment_utils.py#L15 "pdf")

2. Chen P. GridMask data augmentation[J]. arXiv preprint arXiv:2001.04086, 2020.
[pdf](https://arxiv.org/pdf/2001.04086.pdf "pdf") | [github](https://github.com/akuxcw/GridMask "github")

3. Kisantal M, Wojna Z, Murawski J, et al. Augmentation for small object detection[J]. arXiv preprint arXiv:1902.07296, 2019.
[pdf](https://arxiv.org/pdf/1902.07296.pdf "pdf")

## Augmentation zoo for object Detection
### Learning data augmentation strategies for object detection

#### Color Distortion
  - AutoContrast
  - Equalize: Equalize the image histogram
  - Posterize
  - Solarize: Invert all pixels above a threshold value of magniude
  - SolarizeAdd: For each pixel in the image that is less than 128, add an additional amount to it decided by the magnitude.
  - Color: Adjust the color balance of the image.
  - Contrast: Control the contrast of the image.
  - Brightness: Adjust the brightness of the image.
  - Sharpness: Adjust the sharpness of the image
  - Solarize_Only_BBoxes
  - Equalize_Only_Bboxes
  
![ColourDistortion](https://github.com/zzl-pointcloud/Data_Augmentation_Zoo_for_Object_Detection/blob/master/show_img/Color_trans.png)

#### Spatial Transformation
  - Cutout
  - BBox_Cutout
  - Flip
  - Rotate_BBox
  - TranslateX_BBox                                                    
  - TranslateY_BBox                                             
  - ShearX_BBox                                                          
  - ShearY_BBox 
  - TranslateX_Only_BBoxes
  - TranslateY_Only_BBoxes
  - Rotate_Only_BBoxes
  - ShearX_Only_BBoxes
  - ShearY_Only_BBoxes
  - Flip_Only_BBoxes
  - Cutout_Only_Bboxes
  
![SpatialTransformation](https://github.com/zzl-pointcloud/Data_Augmentation_Zoo_for_Object_Detection/blob/master/show_img/Geo_Trans.png)
#### Learned augmentation policy
  - Policy v0, v1, and custom were used in AutoAugment Detection Paper
  - Policy v2, v3 are additional policies that perform well on object detection
  - Policy v4 is the policy which mentioned in this paper, "the best"
  
#### How to use
  
  Make sure the file "/augmentation_zoo/Myautoaugment_utils.py" is in project folder.
  ```python
  from Myautoaugment_utils import AutoAugmenter
  # if you want to use the learned augmentation policy custom or v0-v4(v4 was recommended):
  autoaugmenter = AutoAugmenter('v4')
  # or if you want to use some spatial transformation or color distortion data augmentation，
  # add the data augmentation method that you want to use to the policy_test in Myautoaugment_utils.py 
  # and set the prob and magnitude. For excample:
  # def policy_vtest():
  #    policy = [
  #        [('Color', 0.0, 6), ('Cutout', 0.6, 8)],
  #    ]
  #    return policy
  autoaugmenter = AutoAugmenter('test')
  # Input: 
  #   Sample = {'img': img, 'annot': annots}
  #   img = [H, W, C], RGB, value between [0,1]
  #   annot = [xmin, ymin, xmax, ymax, label]
  # Return:
  #   Sample = {'img': img, 'annot': annots}
  Sample = autoaugmenter(Sample)
  # Use in Pytorch
  dataset = Dataset(root, transform=transforms.Compose([autoaugmenter]))
 ```

### GridMask

![GridMask](https://github.com/zzl-pointcloud/Data_Augmentation_Zoo_for_Object_Detection/blob/master/show_img/GridMask_Trans.png)

Make sure the file "/augmentation_zoo/MyGridMask.py" is in project folder. And the input and output requirements are same as above
```python
from MyGridMask import GridMask
GRID = False
GRID_ROTATE = 1
GRID_OFFSET = 0
GRID_RATIO = 0.5
GRID_MODE = 1
GRID_PROB = 0.5
Gridmask = GridMask(True, True, GRID_ROTATE,GRID_OFFSET,GRID_RATIO,GRID_MODE,GRID_PROB)
Sample = Gridmask(Sample)
```

### Augmentation for small object detection

![SmallobjectAugmentation](https://github.com/zzl-pointcloud/Data_Augmentation_Zoo_for_Object_Detection/blob/master/show_img/Small_Object.png)

This method includes 3 Copy-Pasting Strategies:

1. Pick one small object in an image and copy-paste it 1 time in random locations. 
2. Choose numerous small objects and copy-paste each of these 3 times in an arbitrary position. 
3. Copy-paste all small objects in each image 1 times in random places.

I code it in this way:
```
Algorithm: Augmentation for small object detection
Input: Sample x, Policy v, Threshold thresh, Prob prob
function SmallObjectAugmentation(x, v, thresh, prob)
	Perform the function with the probability of prob
	img, annots = x[‘img’], x[‘annot’]
	for annot in annots do
		if issmallobject(annot, thresh) do
			small_object_list.append(annot)
		end if
	end for
	copy_times and copy_object_num were decided by v
	shuffle the small_object_list
	for idx in range(copy_object_num) do
		to_be_copied_annot = small_object_list[idx]
		for _ in range(copy_times) do
			new_annot = create_copy_annot(to_be_copied_annot, annots)
			if new_annot is not None do
				img = add_patch_in_img(new_annot, to_be_copied_annot, img)
				annots.append(new_annot)
			end if
		end for
	end for
	return {‘img’: img, ‘annot’:annots}
End function 
```
To use this method, make sure the file "/augmentation_zoo/SmallObjectAugmentation.py" is in project folder. And the input and output requirements are same as above
```python
"""   SMALL OBJECT AUGMENTATION   """
# Defaultly perform Policy 2, if you want to use   
# Policy 1, make SOA_ONE_OBJECT = Ture, or if you 
# want to use Policy 3, make SOA_ALL_OBJECTS = True
SOA_THRESH = 64*64
SOA_PROB = 1
SOA_COPY_TIMES = 3
SOA_EPOCHS = 30
SOA_ONE_OBJECT = False
SOA_ALL_OBJECTS = False
augmenter = SmallObjectAugmentation(SOA_THRESH, SOA_PROB, SOA_COPY_TIMES, SOA_EPOCHS, SOA_ALL_OBJECTS, SOA_ONE_OBJECT)
Sample = augmenter(Sample)
```

## Experiment
I use the RetinaNet with ResNet-18, testing in VOC and KITTI. VOC_BATCH_SIZE = 8, KITTI_BATCH_SIZE = 24
| DataSets | No Augmentation | Random Flip | Autoaugmenter('v1') | Autoaugmenter('v4') | GridMask | Small Object Augmentation |
| -------- | --------------- | ----------- | ------------------- | ------------------- | -------- | ------------------------- |
|    VOC   |     0.61492     |   0.63738   |       0.63651       |       0.62267       |  0.65605 |          0.63870          | 
|   KITTI  |     0.60375     |   0.63077   |       0.58631       |       0.64347       |  0.71868 |          0.62622          |

| KITTI | Car | van | truck | pedestrian | Person_sitting | cyclist | Tram | Misc | mAP |
| ----- | --- | --- | ----- | ---------- | -------------- | ------- | ---- | ---- | --- |
| No Augmenation | 0.80197 | 0.64498 | 0.84922 | 0.52134 | 0.27078 | 0.50485 | 0.79602 | 0.47798 | 0.60375 |
| Random Flip | 0.82147 | 0.66679 | 0.85982 | 0.54333 | 0.35177 | 0.53576 | 0.80296 | 0.46428 | 0.63077 |
| AutoAugmenter(‘v1’) | 0.82838 | 0.55684 | 0.74368 | 0.55461 | 0.38256 | 0.48259 | 0.74964 | 0.39217 | 0.58631 |
| AutoAugmenter(‘v4’) | 0.82566 | 0.64851 | 0.87592 | 0.54426 | 0.40881 | 0.56872 | 0.80106 | 0.47480 | 0.64347 |
| GridMask | 0.85529 | 0.75980 | 0.91472 | 0.59351 | 0.51708 | 0.62842 | 0.86256 | 0.61804 | 0.71867 |
| Small Object Augmentation | 0.83064 | 0.60390 | 0.85146 | 0.55325 | 0.42748 | 0.50965 | 0.76821 | 0.46511 | 0.62621 |


## My Contributions
1. Realized the data preprocessing of VOC and KITTI in VocDataset/KittiDataset, prepare_data.py, which could be used by modifying the file path in config.py
2. Realized Augmentation for small object Detection
3. Modified the code of other papers and adjusted it to Numpy format
4. Tested these methods on data sets VOC and KITTI
