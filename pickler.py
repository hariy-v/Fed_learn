import os
import pickle

def pickle_results(folder,server_side_accuracy,server_side_loss,server_side_precision,server_side_recall,server_side_fscore):
    os.makedirs(folder,exist_ok=True)
     
    with open(f'{folder}/server_side_accuracy.pkl', 'wb') as f:
        pickle.dump(server_side_accuracy, f)
        
    with open(f'{folder}/server_side_loss.pkl', 'wb') as f:
        pickle.dump(server_side_loss, f)

    with open(f'{folder}/server_side_precision.pkl', 'wb') as f:
        pickle.dump(server_side_precision, f)
        
    with open(f'{folder}/server_side_recall.pkl', 'wb') as f:
        pickle.dump(server_side_recall, f)

    with open(f'{folder}/server_side_fscore.pkl', 'wb') as f:
        pickle.dump(server_side_fscore, f)