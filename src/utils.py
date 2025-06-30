import pickle

import torch
from scipy.spatial import distance

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def numpy_to_tensor(arr, use_gpu=True, device="cuda:0"):
    if use_gpu:
        return torch.tensor(arr).to(device).float()
    else:
        return torch.tensor(arr).float()


def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def pickle_load(f):
    return pickle.load(open(f, "rb"))


def pickle_dump(obj, f):
    pickle.dump(obj, open(f, "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def load_state_dict(model, pretrained_dict):
    # To bypass nn.DataParallel if needed
    # https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379/15
    try:
        print(model.load_state_dict(pretrained_dict))
    except RuntimeError:
        pretrained_dict = {
            key.replace("module.", ""): value for key, value in pretrained_dict.items()
        }
        print(model.load_state_dict(pretrained_dict))
