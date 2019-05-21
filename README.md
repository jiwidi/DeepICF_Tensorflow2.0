# DeepICF_Tensorflow2.0
A Tensorflow2.0 implementation of the DeepICF 


## Create dataset files

Execute 
```
python create_dataset.py --input_path PATH --verbose 1 --num_neg_test 4 --num_neg_train 4 --force_download 4 --input_path 
```

This will download the data and create it under the PATH variable, we recommend to use data/ as PATH
