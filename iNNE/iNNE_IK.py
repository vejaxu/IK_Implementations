import numpy as np
from random import sample
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
class iNN_IK:
    data = None
    centroid = []
    def __init__(self, psi, t):
        self.psi = psi # size of subsampling
        self.t = t # iterations

    def fit_transform(self, data):
        self.data = data
        self.centroid = [] # centroid should have t elements and each element has psi elements
        self.centroids_radius = [] # centroids_radius's shape is equal to centroid's shape
        sn = self.data.shape[0]
        n, d = self.data.shape
        IDX = np.array([])  # column index
        V = []
        for i in range(self.t):
            subIndex = sample(range(sn), self.psi) # return a list of unique elements randomly chosen from the given range
            print(f"subindex {i}: {subIndex}")
            
            self.centroid.append(subIndex) # its shape is t * psi
            tdata = self.data[subIndex, :]
            print(f"tdata {i}: ")
            print(tdata)
            tt_dis = cdist(tdata, tdata)
            print(f"tt_dis {i}: ")
            print(tt_dis)
            
            # the iteration is to calculate each point's radius
            radius = []
            for r_idx in range(self.psi):
                r = tt_dis[r_idx]
                r[r<0] = 0
                r = np.delete(r,r_idx)
                radius.append(np.min(r))
            print(f"radius i: {radius}")
            
            self.centroids_radius.append(radius) # its shape is t * psi

            # for each point in data, find its nearest neighobur in tdata
            nt_dis = cdist(tdata, self.data)
            print(f"nt_dis {i}: ")
            print(nt_dis)
            centerIdx = np.argmin(nt_dis, axis=0)
            print(f"centerIdx {i}: ")
            print(centerIdx)

            # build a list V where each element is 1 or 0 to represent whether each point in data is within the radius of its nearest neighbor in tdata
            for j in range(n):
                V.append(int(nt_dis[centerIdx[j],j] <= radius[centerIdx[j]])) 
            
            print(f"V {i}: ")
            print(V)

            print()
    
            IDX = np.concatenate((IDX, centerIdx + i * self.psi), axis=0) # unique
        print(f"IDX: {IDX}")

        IDR = np.tile(range(n), self.t) # shape = n * t
        # when n = 5, t = 3, IDR = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

        print(f"IDR: {IDR}")

        #V = np.ones(self.t * n) #value
        ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.t * self.psi))
        return ndata

    def fit(self, data):
        self.data = data
        self.centroid = []
        self.centroids_radius = []
        sn = self.data.shape[0]
        for i in range(self.t):
            subIndex = sample(range(sn), self.psi)
            self.centroid.append(subIndex)
            tdata = self.data[subIndex, :]
            tt_dis = cdist(tdata, tdata)
            radius = [] #restore centroids' radius
            for r_idx in range(self.psi):
                r = tt_dis[r_idx]
                r[r<0] = 0
                r = np.delete(r,r_idx)
                radius.append(np.min(r))
            self.centroids_radius.append(radius)

    def transform(self, newdata):
        assert self.centroid != None, "invoke fit() first!"
        n, d = newdata.shape
        IDX = np.array([])
        V = []
        for i in range(self.t):
            subIndex = self.centroid[i]
            radius = self.centroids_radius[i]
            tdata = self.data[subIndex, :]
            dis = cdist(tdata, newdata)
            centerIdx = np.argmin(dis, axis=0)
            for j in range(n):
                V.append(int(dis[centerIdx[j], j] <= radius[centerIdx[j]]))
            IDX = np.concatenate((IDX, centerIdx + i * self.psi), axis=0)
        IDR = np.tile(range(n), self.t)
        ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.t * self.psi))
        return ndata


if __name__ == '__main__':
    X = np.array([[0, 1],
                  [1, 2], 
                  [2, 3], 
                  [100, 100]])
    
    inne_ik = iNN_IK(3, 10)

    newX = inne_ik.fit_transform(X)
    print(f"newX shape: {newX.shape}")
    print(newX.toarray())

    feature_map = np.dot(newX, newX.T) / newX.shape[0]
    print(f"feature_map shape: {feature_map.shape}")
    print(feature_map.toarray())