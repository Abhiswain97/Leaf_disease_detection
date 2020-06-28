# Leaf disease detection using image segmentation 

This is project on detecting leaf diseases using image segmentation. The catch is I do it without using deep learning. 
Instead I design a multi-stage(3 stage classifier) classifier. 
(Refer to [project_map.png](https://github.com/Abhiswain97/Leaf_disease_detection/blob/master/project_map.png) for details)

### Binary-classification --> Multiclass-classification --> Mask-prediction

So What can you do to use it? <br>

```
git clone https://github.com/Abhiswain97/Leaf_disease_detection.git  
cd Leaf_disease_detection
python run.py --test_image_path <path to your test image>
```

For help: `python run.py -h`

You can also directly run the `Classification.py` in `src` folder. For help: `python Classification.py -h`




