CUB_data_path = '../../CUB_data/CUB_200_2011' # Assume dataset downloaded here

use_cuda = True

seed = 1

iterations = 10
epochs = 100

# Split
n_class = 200
n_class_train = 100
n_class_val = 50

# CUB dataset has 29-30 images per class
samples_per_class = 29

classes_per_it_tr = 50 # train way
num_query_tr = 10 

classes_per_it_val = 5 # val way
num_query_val = 15

cropped_size = 100
batch_size = 16