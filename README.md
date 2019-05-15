### FaceSimilarity 

* Data Preprocessing, Data Augmentation , Modelling and Training is explained in [IPython Notebook](https://github.com/anuj-anuj/face-similarity/blob/master/NanoNets_AnujArora.ipynb)

### Dependencies
* Please follow requirements.txt. Pre-trained models have strict version dependencies on numpy and keras

### Deliverables
1. Use 'predict.py' to get similarity score and binary result of whether two inputs are similar or nor.

* Command Structure: python predict.py image1_path image2_path model_path

* Example Command: python predict.py test_images/Dean_Barkley_0001.jpg test_images/Dean_Barkley_0002.jpg checkpoints/model.15.hd5



### Note:
* Face-Alignment Model and Facenet(Feature-Extraction Model) have been imported with pretrained weights. Sources are mentioned in References


### Refrences 
* FaceNet: A Unified Embedding for Face Recognition and Clustering : https://arxiv.org/abs/1503.03832.
* https://github.com/davidsandberg/facenet
* https://github.com/sainimohit23/FaceNet-Real-Time-face-recognition



