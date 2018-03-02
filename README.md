# tensorflow_triplet
Triplet with AlexNet

Train 3-stream CNNs (triplet) with AlexNet 

Data Preparation:
Put the images of same class into the same directory.
Eg: dogs/dog1.jpg, dogs/dog2.jpg, cats/cat1.jpg...

Training:
python triplet.py --train_dir [training directory] --model_dir [fine-tuned model name] 

Train triplet model and save models.

Testing:
python extract.py --file [img_path] --model_dir [model_path]

Output features of certain image of triplet model.

Reference: Facenet: A unified embedding for face recognition and clustering 
https://arxiv.org/abs/1503.03832
