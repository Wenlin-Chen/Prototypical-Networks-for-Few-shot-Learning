use_cuda = True

seed = 1

classes_per_it_tr = 60 # train way
num_support_tr = 5 # shot = train shot = test shot
num_query_tr = 5

classes_per_it_val = 5 # way
num_support_val = num_support_tr # shot = test shot = train shot
num_query_val = 15

iterations = 100
epochs = 100
learning_rate = 0.001
lr_scheduler_step = 20
lr_scheduler_gamma = 0.5