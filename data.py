import h5py
import os

def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        print(path+' sucess') 
        return True
    else:
        print(path+' already exist') 
        return False

def load_data(in_fn):
    f = h5py.File(in_fn, 'r')
    result = {}
    for key in f.keys():
    	result[key] = f[key][:]
    f.close()
    return result
def save_data(savename, data):
        print('\t\nSaving to {}'.format(savename))
        with h5py.File(savename, "w") as fn:
            for k, v in data.items():
                fn.create_dataset(k, data=v)