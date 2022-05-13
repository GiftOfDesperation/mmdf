# Multi-modal Feature Based Place Recognition
  A deep fusion framework for cross-scene place recognition, this work is accepted by RA-L 2022.  
How to use of our MMDF:
First, you need to install some related libraries such as Pytroch and open3d.   
A requirements.txt can be used to install these libraries conveniently:  
`pip install -r requirements.txt`

Datasets:
Our datasets needs ".pcd" for point clouds, ".png" for images and the poses files.  

The datasets should includes:  
```/pcd  
        /1.pcd  
	/2.pcd  
	...  

and  
/img  
	/1.png  
	/2.png  
	...
```
The file name does not need to be exactly the same as above.  

We construct the datasets for generating sample pairs:  
A example of  generating sample pairs is in utils/generate_pairs.py. You can generate train pairs and test pairs by changing sequence number and the KDTree parameters.  

Once we have the train.txt for sample pairs and the pcd/img file. We can conduct model training:  
Train:  
`python train.py` to train MMDF or MMLF  

Eval:  
We use the thresholds to evaluate the place recognition algorithms.  
eval_on_test_set.py is used to generated a pickle file of distances in a test set.  
eval_thresh is to change the thresholds and generate the precision and recall, so that we can use it to draw the Precsion-Recall curve.  
