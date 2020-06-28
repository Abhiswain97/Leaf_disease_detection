
<br> [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Abhiswain97/Leaf_disease_detection/master)

# Leaf disease detection using image segmentation 

This is project on detecting leaf diseases using image segmentation. The catch is I do it without using deep learning. 
Instead I design a multi-stage(3 stage classifier) classifier. 
(Refer to [project_map.png](https://github.com/Abhiswain97/Leaf_disease_detection/blob/master/project_map.png) for details)

<p align="center">
  <img src="https://github.com/Abhiswain97/Leaf_disease_detection/blob/master/project_map.png" height="500" width="500">
</p>

### Binary-classification --> Multiclass-classification --> Mask-prediction

So What can you do to use it? <br>

To run it with default settings: 

```
git clone https://github.com/Abhiswain97/Leaf_disease_detection.git  
cd Leaf_disease_detection
python run.py --test_image_path <path to your test image>
```

For help: `python run.py -h`

Run it with custom settings: Run the `Classification.py` in `src` folder. 
For help: `python Classification.py -h`




