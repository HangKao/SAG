import numpy as np
from sdataset_geobody import geoDataset
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage as sd


def processAttn(attn):
    attn = np.mean(attn,axis=0)
    attn = np.mean(attn,axis=0)

    attn = attn.reshape((64,64))
    return attn

def processProj(proj):
    proj = proj[0,...]
    proj = np.mean(proj,axis=2)
    return proj


prefix = "../newSamDz/"
seis_train_dataset = geoDataset(prefix,"./testXAI/dzValid.txt")

# sz = seis_train_dataset[0]

# for k in sz.keys():
#     sz[k] = np.array(sz[k])[None,...]

globalIdxArr = [2, 5, 8, 11]

# ep = 49
# ep = 99
# ep = 149
# ep = 199

# ep = 69

for ep in range(9,109,10):

    dataAtten = np.load("./testXAI/out/XAI%dGeobody.npz"%ep,allow_pickle=True)

    print(list(dataAtten.keys()))


    print(len(dataAtten['qkv02']))
    print(dataAtten['qkv02'][0].shape)



    sz = dataAtten['sz'].item()


    sx = sz['image'][0,0,...]
    sx = sx[::2,::2]
    pt = sz['pt'][0,...]/2


    ccmap = plt.cm.Spectral

    nrows,ncols = 2,4


    plt.figure()
    plt.subplot(nrows,ncols,1)
    plt.imshow(sx,cmap='gray')
    plt.scatter(pt[:,0],pt[:,1],c='red')

    for idx in range(1,4):
        
        proj,newproj = dataAtten['proj%02d'%globalIdxArr[idx]]
        proj,newproj = processProj(proj),processProj(newproj)

        print(idx)

        npos = idx+1
        plt.subplot(nrows,ncols,npos)
        # plt.imshow(proj-newproj,cmap='jet')
        # plt.subplot(nrows,ncols,npos+4)
        # plt.imshow(proj,cmap='jet')

        plt.imshow(proj-newproj,cmap=ccmap)
        plt.axis('off')

        plt.subplot(nrows,ncols,npos+4)
        plt.imshow(newproj,cmap=ccmap)
        plt.axis('off')


    # plt.show()
    plt.savefig("./testXAI/%d.png"%ep)






