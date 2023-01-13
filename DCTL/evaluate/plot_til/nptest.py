import numpy as np
def NP():
    width = 256
    height = 256
    npdata = dict()

    data = np.empty((width * height + 1, width, height, 3), dtype="float32")
    print('data shape ',data.shape)
    npdata['npdata'] = data
    print('---')
    np.save('nptest.npy', npdata)
    print('---')
    return  data

if __name__ == '__main__':
   a = NP()
   print(a)