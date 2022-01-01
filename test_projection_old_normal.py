
import numpy as np

def project(forces, friction):
    x = x_in.copy()
    f = forces
    ft = np.sqrt(f[1]**2 + f[2]**2)

    mu = friction[i]
    fn=min(f[0], self.const_Normals[i])
    if ft <= mu *fn:
        if(f[0]>self.const_Normals[i] and self.const_Normals[i]>0):
            ratio=self.const_Normals[i]/f[0]
            f[1]*=ratio
            f[2]*=ratio
        continue
    if mu != 0.0 and ft < -f[0] / mu:
        f[0:3] = np.zeros(3)
    elif ft > mu * f[0]:
        currProjectedN= (f[0] + mu * ft) / (mu**2 + 1)
        fn=min(currProjectedN, self.const_Normals[i])
        # if( currProjectedN > self.const_Normals[i] and self.const_Normals[i]>1e-8):
            # ratio=self.const_Normals[i]/currProjectedN
            # ft*=ratio
            # fn=self.const_Normals[i]
            # ft = np.sqrt(f[1]**2 + f[2]**2)
        # note that we don't want to cap the normal force of the current step
        if(ft==0):
            print(currProjectedN, ratio, self.const_Normals[i], f)
        f[1] = f[1] * mu * fn / ft
        f[2] = f[2] * mu * fn / ft
