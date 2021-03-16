import torch

class PrototypicalLoss(torch.nn.Module):
    def __init__(self, device):
        super(PrototypicalLoss, self).__init__()
        self.device = device

    def forward(self, input, target, n_support):
        return self.prototypical_loss(input, target, n_support)

    def euclidean_dist(self, x1, x2):
        """
        Compute Euclidean distance (p=2) between two tensors x1, x2
        x1 shape:    B x P x M
        x2 shape:    B x R x M
        output shape:B x P x R
        """
        return torch.cdist(x1, x2, p=2)

    def prototypical_loss(self, input, target, n_support):
        """
        Average support samples to get prototype center and compute batch's
        loss w.r.t Euclidean distance between query samples and centers
        input:      x-embedding of shape [batch_size, 64], 
                    batch_size = (n_support + n_query) * n_class
        target:     class labels for each input point in the batch
        n_support:  number of examples per class
        """

        # Extract unique classes and initialise variables
        # input_to_class_idx maps each input to one of unique_classes 
        # by index 0 to n_class - 1 (instead of dataset's class numbering)
        unique_classes, input_to_class_idx = torch.unique(target, sorted=False,
                                                        return_inverse=True)
        n_class = len(unique_classes)
        query_list = []
        prototype_list = []

        for k in range(n_class):
            # Get input indices corresponding to class k, flatten to vector
            # input_idx shape: [10]
            input_idx = torch.flatten((input_to_class_idx == k).nonzero(
                                                        as_tuple = False))
            
            # Extract inputs corresponding to class k
            # input_k shape: [n_class, 64]
            input_k = torch.index_select(input, 0, input_idx)
            
            # Separate support and query vectors
            # support_k shape: [n_support, 64]
            # query_k shape: [n_query, 64]
            support_k = input_k[:n_support]
            query_k = input_k[n_support:]

            # Average over support inputs of class k to get prototype
            # prototype_k shape: [64]
            prototype_k = torch.mean(support_k, 0)
            
            # Save
            query_list.append(query_k)
            prototype_list.append(prototype_k)

        # Stack classes on top of each other
        # query shape: [n_class, n_query, 64]
        query = torch.stack(query_list)
        # prototype shape: [n_class, 64]
        prototype = torch.stack(prototype_list)

        # Make query target with index from 0 to n_class-1 
        # query_target shape (matches log_p_y): [n_class, n_class, n_query]
        # query_target2 shape (matches y_hat): [n_class, n_query]
        n_query = query_list[0].size()[0]
        query_target = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_class, n_query).to(self.device)
        query_target2 = torch.arange(0, n_class).view(n_class, 1).expand(n_class, n_query).to(self.device)

        # Compute distances between each class' queries and each class' 
        # prototype 
        # dists shape: [n_class, n_class, n_query] 
        dists = self.euclidean_dist(prototype, query)

        # Pass negative distances through softmax to give log likelihood 
        # log(p(y=k|x))
        # log_p_y shape: [n_class, n_class, n_query] 
        log_p_y = torch.nn.functional.softmax(-dists, dim=1)

        # Predicted class
        # y_hat shape: [n_class, n_query]
        _, y_hat = log_p_y.max(dim=1)

        '''
        print("""query: \t \t {} \n 
            prototype: \t \t {} \n 
            dists: \t \t {} \n 
            query_target: \t {} \n 
            log_p_y: \t \t {} \n 
            query_target2: \t {} \n 
            y_hat: \t \t {}""".format(query.size(), 
            prototype.size(), dists.size(),
            query_target.size(), log_p_y.size(),
            query_target2.size(), y_hat.size()))
        '''

        # Compute loss and accuracy
        # Loss is the average negative log likelihoods for target class
        # Accuracy is the average number of correct classification
        loss_val = -log_p_y.gather(1, query_target).mean()
        acc_val = y_hat.eq(query_target2).float().mean()
        #print("loss_val = {} \n accuracy_val = {}".format(loss_val, acc_val))
        #print("---------------------------------------------------------")

        return loss_val, acc_val