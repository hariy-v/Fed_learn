import numpy as np
from scipy.stats import dirichlet

# def allocate_images(dirichlet_proportions, total_images_per_client):
#     """
#     Allocates images to clients based on normalized Dirichlet proportions,
#     ensuring each client receives a specified total number of images.

#     :param dirichlet_proportions: Proportions from the Dirichlet distribution.
#     :param total_images_per_client: Total images each client should receive.
#     :return: Allocation matrix with the number of images for each class per client.
#     """
#     # Normalize proportions for each client
#     normalized_proportions = dirichlet_proportions / dirichlet_proportions.sum(axis=0)

#     # Allocate images based on normalized proportions
#     allocations = normalized_proportions * total_images_per_client

#     return allocations.astype(int)

# # Example usage
# alpha = 0.5
# num_classes = 3
# num_clients = 3
# num_images = 300
# total_images_per_client = num_images // num_clients

# # Generate Dirichlet proportions
# dirichlet_proportions = dirichlet.rvs([alpha] * num_clients, size=num_classes)

# print(dirichlet_proportions)

# # Allocate images
# allocations = allocate_images(dirichlet_proportions, total_images_per_client)
# print("Allocations:\n", allocations)

# print(np.sum(allocations ,axis = 1))

y_train = np.array([i for i in range(10) for _ in range(100)])
alpha = 0.5
num_classes = 10
num_clients = 10

def dirichlet_allocation_balanced(alpha, num_classes, num_clients):
    """
    Allocate indices of trainset among clients based on normalized Dirichlet distribution,
    ensuring each client receives a specified number of images.

    :param trainset: Dataset with labels.
    :param alpha: Concentration parameter for Dirichlet distribution.
    :param num_classes: Number of classes in the dataset.
    :param num_clients: Number of clients.
    :param images_per_client: Number of images each client should receive.
    :return: Dictionary mapping client IDs to their allocated indices.
    """
    # try:
    #     labels = np.array(trainset.targets)
    # except AttributeError:
    #     labels = np.array(trainset.labels)

    labels = np.array([i for i in range(10) for _ in range(100)])
    print(len(labels))

    images_per_client = len(labels) // num_clients
    
    # Initialize client indices storage
    client_indices = {i: [] for i in range(num_clients)}

    # Generate Dirichlet distribution proportions
    dirichlet_proportions = dirichlet.rvs([alpha] * num_clients, size=num_classes)

    # Normalize Dirichlet proportions for each class across clients

    print(dirichlet_proportions)

    normalized_proportions = dirichlet_proportions / np.sum(dirichlet_proportions, axis=0, keepdims=False)

    print(normalized_proportions)

    # Allocate indices based on normalized proportions
    for k in range(num_classes):
        class_k_indices = np.where(labels == k)[0]
        np.random.shuffle(class_k_indices)  # Shuffle indices to ensure random allocation
        
        # Calculate the exact number of images to allocate per client for this class
        allocations = (normalized_proportions[k] * images_per_client).astype(int)
        
        start = 0
        for i in range(num_clients):
            end = start + allocations[i]
            client_indices[i].extend(class_k_indices[start:end])
            start = end

    # Convert lists to arrays
    for client_id in client_indices:
        client_indices[client_id] = np.array(client_indices[client_id])

    return client_indices


a = dirichlet_allocation_balanced( alpha, num_classes, num_clients)

print(a)

for k in a:
    print(len(a[k]))

NUM_CLIENTS = 10
ALPHA = 0.5

# def CIFAR10_SuperClass_NIID_DIR():

#     '''
#     This function creates Dirichlet Non-IID split of CIFAR10 dataset based on Super clusters

#     Parameters:
#         NUM_CLIENTS(int) : Number of clients.
#         alpha(float)     : Parameter in Dirichlet distribution

#     returns:
#         net_dataidx_map(dict)      : Gives indices of train images assigned to each client.
#         net_dataidx_map_test(dict) : Gives indices of test images assigned to each client.
#         traindata_cls_counts(dict) : Gives count of train labels at each client.
#         testdata_cls_counts(dict)  : Gives count of test labels at each client.

 
 
#     '''
#     y_train = []
#     for i in range(10):
#         y_train+=[i for _ in range(100)]

#     y_train = np.array(y_train)

#     y_test = []
#     for i in range(10):
#         y_test+=[i for _ in range(100)]

#     y_test = np.array(y_test)



#     superclass = [[0,1,8,9], [2,3,4,5,6,7]]
#     nclass=10
#     idxs_superclass = {}
#     net_dataidx_map = {i:np.array([],dtype='int') for i in range(NUM_CLIENTS)}
#     net_dataidx_map_test = {i:np.array([],dtype='int') for i in range(NUM_CLIENTS)}
#     traindata_cls_counts = {}
#     testdata_cls_counts = {}
#     cnt=0

#     n_parties_ratio = np.array([1/len(superclass) for i in range(len(superclass))])
#     for i in range(len(superclass)):
#         n_parties_ratio[i]+= n_parties_ratio[i]*(len(superclass[i])/nclass)

#     n_parties_ratio = n_parties_ratio/sum(n_parties_ratio)
#     n_parties_ratio = [int(np.ceil(el*NUM_CLIENTS)) for el in n_parties_ratio]
#     s = sum(n_parties_ratio)
#     if s>NUM_CLIENTS:
#         inds = np.random.choice(len(n_parties_ratio), size=s-NUM_CLIENTS, replace=True)
#         for _i in inds:
#             n_parties_ratio[_i]-=1
#     elif s<NUM_CLIENTS:
#         inds = np.random.choice(len(n_parties_ratio), size=NUM_CLIENTS-s, replace=True)
#         for _i in inds:
#             n_parties_ratio[_i]+=1

#     assert sum(n_parties_ratio)==NUM_CLIENTS

#     for r, clust in enumerate(superclass):
#         ##### Forming the labels for each clients
#         #n_parties=int(len(clust)/nclass*NUM_CLIENTS)
#         n_parties = n_parties_ratio[r]
#         N=int(len(clust)*5000)

#         min_size = 0
#         min_require_size = 15
#         #beta = 0.1
#         #np.random.seed(2021)
#         print(clust)
#         while min_size < min_require_size:
#             idx_batch = [[] for _ in range(n_parties)]
#             for k in clust:
#                 idx_k = np.where(y_train == k)[0]
#                 np.random.shuffle(idx_k)

#                 proportions = np.random.dirichlet(np.repeat(ALPHA, n_parties))
#                 proportions = proportions / proportions.sum()
#                 proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
#                 proportions = proportions / proportions.sum()
#                 proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

#                 idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
#             #print(sum([len(idx_j) for idx_j in idx_batch]))
#             min_size = min([len(idx_j) for idx_j in idx_batch])

#         #### Assigning samples to each client
#         for j in range(cnt, cnt+n_parties):
#             np.random.shuffle(idx_batch[j-cnt])
#             net_dataidx_map[j] = np.hstack([net_dataidx_map[j], idx_batch[j-cnt]])

#             dataidx = net_dataidx_map[j]
#             unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
#             tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
#             traindata_cls_counts[j] = tmp

#             for key in tmp.keys():
#                 idxs_test = np.where(y_test==key)[0]
#                 net_dataidx_map_test[j] = np.hstack([net_dataidx_map_test[j], idxs_test])

#             dataidx = net_dataidx_map_test[j]
#             unq, unq_cnt = np.unique(y_test[dataidx], return_counts=True)
#             tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
#             testdata_cls_counts[j] = tmp

#         cnt+=n_parties


import numpy as np

# def CIFAR10_SuperClass_NIID_DIR():
#     '''
#     Modified function to create Dirichlet Non-IID split of CIFAR10 dataset based on superclusters,
#     ensuring normalized allocation per client.

#     Parameters:
#         NUM_CLIENTS(int): Number of clients.
#         ALPHA(float): Parameter in Dirichlet distribution.

#     Returns:
#         net_dataidx_map(dict): Indices of train images assigned to each client.
#         net_dataidx_map_test(dict): Indices of test images assigned to each client.
#         traindata_cls_counts(dict): Count of train labels at each client.
#         testdata_cls_counts(dict): Count of test labels at each client.
#     '''

#     # Assuming y_train and y_test are predefined arrays of labels for training and testing datasets
#     y_train = np.array([i for i in range(10) for _ in range(1000)])
#     y_test = np.array([i for i in range(10) for _ in range(1000)])

#     superclass = [[0,1,8,9], [2,3,4,5,6,7]]
#     net_dataidx_map = {i: np.array([], dtype='int') for i in range(NUM_CLIENTS)}
#     net_dataidx_map_test = {i: np.array([], dtype='int') for i in range(NUM_CLIENTS)}
#     traindata_cls_counts = {}
#     testdata_cls_counts = {}

#     for r, clust in enumerate(superclass):
#         # Directly using the modified allocation logic for simplicity
#         n_parties = NUM_CLIENTS // len(superclass)  # Simplified allocation for illustration
#         for k in clust:
#             idx_k = np.where(y_train == k)[0]
#             np.random.shuffle(idx_k)

#             # Generate and normalize Dirichlet proportions
#             proportions = np.random.dirichlet(np.repeat(ALPHA, n_parties))
#             proportions /= proportions.sum()  # Ensure sum of proportions equals 1

#             # Determine the number of images per client based on normalized proportions
#             num_images = len(idx_k)
#             images_per_client = [int(proportion * num_images) for proportion in proportions]

#             # Adjust for rounding errors to ensure total allocation matches num_images
#             difference = num_images - sum(images_per_client)
#             for i in range(abs(difference)):
#                 images_per_client[i % len(images_per_client)] += np.sign(difference)

#             # Allocate images to clients
#             start = 0
#             for i, num_images in enumerate(images_per_client):
#                 end = start + num_images
#                 client_id = r * (NUM_CLIENTS // len(superclass)) + i  # Adjust client ID based on superclass
#                 net_dataidx_map[client_id] = np.concatenate((net_dataidx_map[client_id], idx_k[start:end]))
#                 start = end

# #     # The rest of the function remains unchanged, focusing on test data allocation and counting class instances
# #     for i in (net_dataidx_map):
# #         print(i)
# #         print(len(net_dataidx_map[i]))
# #     return net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts

# import numpy as np
# from scipy.stats import dirichlet

# def CIFAR10_SuperClass_NIID_DIR():
#     '''
#     Allocate CIFAR10 dataset indices among clients based on superclusters and Dirichlet distribution.

#     Parameters:
#         alpha(float): Parameter in Dirichlet distribution.
#         NUM_CLIENTS(int): Total number of clients, default is 10.

#     Returns:
#         net_dataidx_map: Indices of train images assigned to each client.
#     '''
#     # Example data setup
#     y_train = np.array([i for i in range(10) for _ in range(100)])  # Simplified label distribution
#     print(y_train)
#     print(len(y_train))
#     # Define superclusters
#     superclass = [[0, 1, 8, 9], [2, 3, 4, 5, 6, 7]]
#     net_dataidx_map = {i: np.array([], dtype='int') for i in range(NUM_CLIENTS)}

#     # Allocation for each supercluster
#     for supercluster_id, clust in enumerate(superclass):
#         # Select clients for this supercluster
#         clients = range(supercluster_id * (NUM_CLIENTS // 2), (supercluster_id + 1) * (NUM_CLIENTS // 2))
        
#         # Calculate total images for this supercluster
#         total_images = sum([np.sum(y_train == k) for k in clust])
        
#         # Generate Dirichlet distribution for clients in this supercluster
#         dirichlet_proportions = dirichlet.rvs([ALPHA] * len(clients))
        
#         # Normalize the proportions so they sum up to the total images
#         normalized_proportions = (dirichlet_proportions / dirichlet_proportions.sum()) * total_images
        
#         start_index = 0
#         for client_id, proportion in zip(clients, normalized_proportions.ravel()):
#             for k in clust:
#                 # Find indices for class k
#                 idx_k = np.where(y_train == k)[0]
#                 np.random.shuffle(idx_k)

#                 # Calculate number of images from class k for this client based on proportion
#                 num_images_k = int(proportion * len(idx_k) / total_images)
                
#                 # Assign images to client
#                 end_index = start_index + num_images_k
#                 net_dataidx_map[client_id] = np.concatenate((net_dataidx_map[client_id], idx_k[start_index:end_index]))
                
#                 start_index = end_index if end_index < len(idx_k) else 0

#     for i in net_dataidx_map:
#         print(i)
#         print(len(net_dataidx_map[i]))
    
#     return net_dataidx_map


# CIFAR10_SuperClass_NIID_DIR()


# # Create dummy y_train
# num_classes = 10
# data_points_per_class = 100
# y_train = np.array([i for i in range(num_classes) for _ in range(data_points_per_class)])

# # Define superclasses
# superclass = [[0, 1, 8, 9], [2, 3, 4, 5, 6, 7]]

# # Assign clients to each superclass
# num_clients = 10
# clients_per_group = num_clients // len(superclass)  # 5 clients per group

# from scipy.stats import dirichlet

# def allocate_data(y_train, superclass, alpha, total_clients):
#     data_per_client = len(y_train) // (len(superclass) * total_clients)
#     allocations = {i: [] for i in range(total_clients)}
#     client_data_counts = {i: 0 for i in range(total_clients)}

#     for cls in superclass:
#         class_indices = np.where(y_train == cls)[0]
#         np.random.shuffle(class_indices)
        
#         # Generate Dirichlet distribution for this class
#         proportions = dirichlet.rvs([alpha] * total_clients)[0]
#         proportions = proportions / proportions.sum()  # Normalize
        
#         # Calculate the number of datapoints per client for this class
#         data_counts = np.floor(proportions * data_per_client * len(superclass)).astype(int)
        
#         # Adjust to ensure total count is met exactly
#         deficit = len(class_indices) - data_counts.sum()
#         data_counts[np.argsort(data_counts)[:deficit]] += 1
        
#         start = 0
#         for i in range(total_clients):
#             end = start + data_counts[i]
#             allocations[i].extend(class_indices[start:end])
#             client_data_counts[i] += data_counts[i]
#             start = end

#     return allocations, client_data_counts

# # Allocate data for each superclass group
# allocations = {}
# for i, sc in enumerate(superclass):
#     start_client_id = i * clients_per_group
#     sc_allocations, _ = allocate_data(y_train, sc, 0.5, clients_per_group)
#     for j in range(clients_per_group):
#         allocations[start_client_id + j] = sc_allocations[j]

# # Count classes in allocations
# def count_classes(client_allocations, y_train):
#     class_counts = {client: {cls: 0 for cls in range(num_classes)} for client in client_allocations}
#     for client, indices in client_allocations.items():
#         for idx in indices:
#             cls = y_train[idx]
#             class_counts[client][cls] += 1
#     return class_counts

# class_counts = count_classes(allocations, y_train)

# # Display the counts
# for client, counts in class_counts.items():
#     print(f"Client {client}: {counts}")

