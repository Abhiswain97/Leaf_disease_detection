# Leaf disease detection using image segmentation 

This is project on detecting leaf diseases using image segmentation. The catch is I do it without using deep learning. 
Instead I design a multi-stage(3 stage classifier) classifier. 
(Refer to [project_map.png](https://github.com/Abhiswain97/Leaf_disease_detection/blob/master/project_map.png) for details)

### Binary-classification --> Multiclass-classification --> Mask-prediction

So What can you do to use it?
Currently, I have created the feature files(made using glcm features), U can run the classification(Binary and multiclass) and reproduce my results. Do:
`python Classification.py --model <model-name>`


It's a work in progress.... So there's still polishing going on I will keep it coming!
