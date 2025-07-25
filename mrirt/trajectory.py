
import numpy as np
import math

class Trajectory(object):

    def __init__(self,applied_timesteps=None,**kwargs):
        self.paramDict=kwargs
        self.traj = None
        self.traj_for_reconstruction=None
        self.applied_timesteps=applied_timesteps
        self.reconstruct_each_partition = False #For 3D - whether all reps are used for generating the kspace data or only the current partition


    def get_traj(self):
        #Returns and stores the trajectory array of ntimesteps * total number of points * ndim
        raise ValueError("get_traj should be implemented in child")

    def get_traj_for_reconstruction(self,timesteps=175):
        if self.traj_for_reconstruction is not None:
            print("Warning : Outputting the stored reconstruction traj - timesteps input has no impact - please reset with self.traj_for_reconstruction=None")
            return self.traj_for_reconstruction

        else:
            traj = self.get_traj()
            #try:
            return traj.reshape(timesteps,-1,traj.shape[-1])
            #except:
            #    window=np.ceil(self.paramDict["total_nspokes"]/timesteps)+1
            #    traj_reco= np.array(groupby(traj,window))
            #    return traj_reco.reshape(timesteps,-1,traj_reco.shape[-1])

    def adjust_traj_for_window(self,window):
        traj=self.get_traj()
        traj_shape=traj.shape
        traj=np.array(groupby(traj,window))
        traj=traj.reshape((-1,)+traj_shape[1:])
        self.traj=traj

class Radial(Trajectory):

    def __init__(self,total_nspokes=1400,npoint=512,**kwargs):
        super().__init__(**kwargs)

        self.paramDict["total_nspokes"]=total_nspokes #total nspokes per rep
        self.paramDict["npoint"] = npoint
        self.paramDict["nb_rep"]=1

    def get_traj(self):
        if self.traj is None:
            npoint = self.paramDict["npoint"]
            total_nspokes = self.paramDict["total_nspokes"]
            all_spokes = radial_golden_angle_traj(total_nspokes, npoint)
            #traj = np.reshape(groupby(all_spokes, nspoke), (-1, npoint * nspoke))
            traj = all_spokes
            traj=np.stack([traj.real,traj.imag],axis=-1)
            self.traj=traj

        return self.traj


class Cartesian(Trajectory):

    def __init__(self,total_nspokes=1400,npoint_x=256,npoint_y=1,**kwargs):
        super().__init__(**kwargs)

        self.paramDict["total_nspokes"]=total_nspokes #total nspokes per rep
        self.paramDict["npoint_x"] = npoint_x
        self.paramDict["npoint_y"] = npoint_y

        self.paramDict["nb_rep"]=1

    def get_traj(self):
        if self.traj is None:
            npoint_x = self.paramDict["npoint_x"]
            npoint_y = self.paramDict["npoint_y"]
            total_nspokes = self.paramDict["total_nspokes"]

            base_traj=cartesian_traj_2D(npoint_x,npoint_y)

            #traj = np.reshape(groupby(all_spokes, nspoke), (-1, npoint * nspoke))
            traj = np.tile(base_traj,(total_nspokes,1,1))
            self.traj=traj

        return self.traj





class Radial3D(Trajectory):

    def __init__(self,total_nspokes=1400,nspoke_per_z_encoding=8,npoint=512,undersampling_factor=1,incoherent=False,is_random=False,mode="old",offset=0,golden_angle=True,nb_rep_center_part=1,**kwargs):
        super().__init__(**kwargs)
        self.paramDict["total_nspokes"] = total_nspokes
        self.paramDict["nspoke"] = nspoke_per_z_encoding
        self.paramDict["npoint"] = npoint
        self.paramDict["undersampling_factor"] = undersampling_factor
        self.paramDict["nb_rep"]=math.ceil(self.paramDict["nb_slices"]/self.paramDict["undersampling_factor"])
        print(self.paramDict["nb_rep"])
        self.paramDict["random"]=is_random
        self.paramDict["incoherent"]=incoherent
        self.paramDict["mode"] = mode
        if self.paramDict["mode"]=="Kushball":
            self.paramDict["incoherent"]=True
        
        self.paramDict["offset"] = offset
        self.paramDict["golden_angle"]=golden_angle
        self.paramDict["nb_rep_center_part"] = nb_rep_center_part


    def get_traj(self):
        if self.traj is None:
            nspoke = self.paramDict["nspoke"]
            npoint = self.paramDict["npoint"]
            mode = self.paramDict["mode"]
            offset=self.paramDict["offset"]

            total_nspokes = self.paramDict["total_nspokes"]
            nb_slices=self.paramDict["nb_slices"]
            undersampling_factor=self.paramDict["undersampling_factor"]

            nb_rep_center_part=self.paramDict["nb_rep_center_part"]

            if self.paramDict["golden_angle"]:
                if self.paramDict["mode"]=="Kushball":
                    self.traj=self.traj=spherical_golden_angle_means_traj_3D(total_nspokes, npoint, nb_slices,undersampling_factor)

                elif self.paramDict["random"]:
                    if "frac_center" in self.paramDict:
                        self.traj = radial_golden_angle_traj_random_3D(total_nspokes, npoint, nspoke, nb_slices, undersampling_factor,self.paramDict["frac_center"],self.paramDict["mode"],self.paramDict["incoherent"])
                    else:
                        self.traj = radial_golden_angle_traj_random_3D(total_nspokes, npoint, nspoke, nb_slices,
                                                                       undersampling_factor,0.25,self.paramDict["mode"],self.paramDict["incoherent"])
                else:
                    if self.paramDict["incoherent"]:
                        self.traj=radial_golden_angle_traj_3D_incoherent(total_nspokes, npoint, nspoke, nb_slices, undersampling_factor,mode,offset)
                    else:
                        self.traj = radial_golden_angle_traj_3D(total_nspokes, npoint, nspoke, nb_slices,
                                                                undersampling_factor,nb_rep_center_part)

            else:
                self.traj=distrib_angle_traj_3D(total_nspokes, npoint, nspoke, nb_slices,
                                                                undersampling_factor)


        return self.traj



class Spherical3D(Trajectory):

    def __init__(self,total_nspokes=1400,npoint=512,undersampling_factor=4,**kwargs):
        super().__init__(**kwargs)
        self.paramDict["total_nspokes"] = total_nspokes
        self.paramDict["npoint"] = npoint
        self.paramDict["undersampling_factor"] = undersampling_factor
        self.paramDict["nb_rep"]=int(self.paramDict["nb_slices"]/self.paramDict["undersampling_factor"])



    def get_traj(self):
        if self.traj is None:
            #nspoke = self.paramDict["nspoke"]
            npoint = self.paramDict["npoint"]


            total_nspokes = self.paramDict["total_nspokes"]
            nb_slices=self.paramDict["nb_slices"]
            undersampling_factor=self.paramDict["undersampling_factor"]
            self.traj=spherical_golden_angle_means_traj_3D(total_nspokes, npoint, nb_slices,undersampling_factor)


        return self.traj

class Cartesian3D(Trajectory):

    def __init__(self,total_nspokes=1400,npoint_x=256,npoint_y=256,npoint_z=16,**kwargs):
        super().__init__(**kwargs)

        if self.applied_timesteps is not None:
            self.paramDict["total_nspokes"] = len(self.applied_timesteps)  # total nspokes per rep

        else:
            self.paramDict["total_nspokes"]=total_nspokes #total nspokes per rep

        self.paramDict["npoint_x"] = npoint_x
        self.paramDict["npoint_y"] = npoint_y
        self.paramDict["npoint_z"] = npoint_z

        if "reconstruct_each_partition" in self.paramDict:#Navigator - full k-space sampled at each rep
            self.reconstruct_each_partition = self.paramDict["reconstruct_each_partition"]
            if self.paramDict["reconstruct_each_partition"]:
                self.paramDict["npoint"]=npoint_x*npoint_y*npoint_z

        self.paramDict["nb_rep"]=1

    def get_traj(self):
        if self.traj is None:
            npoint_x = self.paramDict["npoint_x"]
            npoint_y = self.paramDict["npoint_y"]
            npoint_z = self.paramDict["npoint_z"]
            total_nspokes = self.paramDict["total_nspokes"]

            #base_traj=cartesian_traj_2D(npoint_x,npoint_y)
            k_max=np.pi
            kx = -k_max + np.arange(npoint_x) * 2 * k_max / (npoint_x - 1)
            ky = -k_max + np.arange(npoint_y) * 2 * k_max / (npoint_y - 1)
            kz = -k_max + np.arange(npoint_z) * 2 * k_max / (npoint_z - 1)

            KX, KY,KZ = np.meshgrid(kx, ky,kz)
            base_traj=np.stack([KX.flatten(), KY.flatten(),KZ.flatten()], axis=-1)

            #traj = np.reshape(groupby(all_spokes, nspoke), (-1, npoint * nspoke))
            traj = np.tile(base_traj,(total_nspokes,1,1))
            self.traj=traj

        return self.traj





class VariableSpiral(Trajectory):

    def __init__(self,total_nspokes=1400,max_gradient=80 * 10 ** -3,max_slew=200,alpha=128,npoint=256,ninterleaves=1,spatial_us=32,temporal_us=0.1,**kwargs):
        super().__init__(**kwargs)
        self.paramDict["total_nspokes"] = total_nspokes
        #self.paramDict["nspiral"] = nspiral #corresponds to the number of spirals by timestep
        self.paramDict["max_gradient"] = max_gradient
        self.paramDict["max_slew"] = max_slew
        self.paramDict["alpha"] = alpha
        self.paramDict["npoint"] = npoint
        self.paramDict["ninterleaves"] = ninterleaves
        self.paramDict["spatial_us"] = spatial_us
        self.paramDict["temporal_us"] = temporal_us

    def get_traj(self):
        if self.traj is None:
            gm = self.paramDict["max_gradient"]
            sm = self.paramDict["max_slew"]
            alpha = self.paramDict["alpha"]
            N = self.paramDict["npoint"]
            fov = N / (2*np.pi)
            ninterleaves = self.paramDict["ninterleaves"]
            R = self.paramDict["spatial_us"]
            f_sampling = self.paramDict["temporal_us"]

            #nspiral = self.paramDict["nspiral"]
            #ntimesteps = self.paramDict["ntimesteps"]

            total_spirals =self.paramDict["total_nspokes"]
            all_spirals = spiral_golden_angle_traj(total_spirals, fov, N, f_sampling, R, ninterleaves, alpha, gm, sm)
            #all_spirals=spiral_golden_angle_traj_v2(total_spirals,nspiral, fov, N, f_sampling, R, ninterleaves, alpha, gm, sm)
            #traj = np.reshape(groupby(all_spirals, nspiral), (ntimesteps, -1))
            traj=np.stack([all_spirals.real, all_spirals.imag], axis=-1)
            self.traj = traj

        return self.traj


class Navigator3D(Trajectory):

    def __init__(self,direction=[0.0,0.0,1.0],npoint=512,nb_slices=1,undersampling_factor=1,nb_gating_spokes=50,**kwargs):
        super().__init__(**kwargs)
        self.paramDict["total_nspokes"] = nb_gating_spokes
        self.paramDict["npoint"] = npoint
        self.paramDict["direction"] = direction
        self.paramDict["nb_slices"] = nb_slices
        self.paramDict["undersampling_factor"] = undersampling_factor
        self.paramDict["nb_rep"] = int(self.paramDict["nb_slices"] / self.paramDict["undersampling_factor"])
        self.reconstruct_each_partition=True

    def get_traj(self):
        if self.traj is None:
            npoint = self.paramDict["npoint"]
            direction=self.paramDict["direction"]
            #nb_rep=self.paramDict["nb_rep"]
            total_nspoke=self.paramDict["total_nspokes"]
            k_max=np.pi

            base_spoke=(-k_max+np.arange(npoint)*2*k_max/(npoint-1)).reshape(-1,1)*np.array(direction).reshape(1,-1)
            self.traj=np.repeat(np.expand_dims(base_spoke,axis=0),axis=0,repeats=total_nspoke)


        return self.traj
    


def cartesian_traj_2D(npoint_x,npoint_y,k_max=np.pi):
    #kx = -k_max + np.arange(npoint_x) * 2 * k_max / (npoint_x - 1)
    #ky = -k_max + np.arange(npoint_y) * 2 * k_max / (npoint_y - 1)
    kx = np.arange(-k_max, k_max, 2 * k_max / npoint_x) + k_max / npoint_x
    ky = np.arange(-k_max, k_max, 2 * k_max / npoint_y) + k_max / npoint_y

    KX, KY = np.meshgrid(kx, ky)
    return np.stack([KX.flatten(), KY.flatten()], axis=-1)

def cartesian_traj_3D(total_nspoke, npoint_x, npoint_y, nb_slices, undersampling_factor=4):
    timesteps = int(total_nspoke / nspoke)
    nb_rep = int(nb_slices / undersampling_factor)
    base_traj=cartesian_traj_2D(npoint_x,npoint_y)
    traj = np.tile(base_traj, (total_nspoke, 1, 1))

    k_z = np.zeros((timesteps, nb_rep))
    all_slices = np.linspace(-np.pi, np.pi, nb_slices)
    k_z[0, :] = all_slices[::undersampling_factor]
    for j in range(1, k_z.shape[0]):
        k_z[j, :] = np.sort(np.roll(all_slices, -j)[::undersampling_factor])




def radial_golden_angle_traj(total_nspoke,npoint,k_max=np.pi):
    golden_angle=111.246*np.pi/180
    #base_spoke = np.arange(-k_max, k_max, 2 * k_max / npoint, dtype=np.complex_)
    #base_spoke = -k_max+np.arange(npoint)*2*k_max/(npoint-1)
    base_spoke = (-k_max+k_max/(npoint)+np.arange(npoint)*2*k_max/(npoint))
    all_rotations = np.exp(1j * np.arange(total_nspoke) * golden_angle)
    all_spokes = np.matmul(np.diag(all_rotations), np.repeat(base_spoke.reshape(1, -1), total_nspoke, axis=0))
    return all_spokes

def distrib_angle_traj(total_nspoke,npoint,k_max=np.pi):
    angle=2*np.pi/total_nspoke
    #base_spoke = np.arange(-k_max, k_max, 2 * k_max / npoint, dtype=np.complex_)
    base_spoke = -k_max+np.arange(npoint)*2*k_max/(npoint-1)
    all_rotations = np.exp(1j * np.arange(total_nspoke) * angle)
    all_spokes = np.matmul(np.diag(all_rotations), np.repeat(base_spoke.reshape(1, -1), total_nspoke, axis=0))
    return all_spokes


def radial_golden_angle_traj_3D(total_nspoke, npoint, nspoke, nb_slices, undersampling_factor=4,nb_rep_center_part=1):
    timesteps = int(total_nspoke / nspoke)
    print(total_nspoke)
    print(nspoke)


    nb_rep = math.ceil((nb_slices ) / undersampling_factor)+nb_rep_center_part-1
    all_spokes = radial_golden_angle_traj(total_nspoke, npoint)
    #traj = np.reshape(all_spokes, (-1, nspoke * npoint))

    k_z = np.zeros((timesteps, nb_rep))
    #all_slices = np.linspace(-np.pi, np.pi, nb_slices)
    all_slices=np.arange(-np.pi, np.pi, 2 * np.pi / nb_slices)
    

    k_z[0, :] = all_slices[::undersampling_factor]


    for j in range(1, k_z.shape[0]):
        k_z[j, :] = np.sort(np.roll(all_slices, -j)[::undersampling_factor])

    if nb_rep_center_part>1:
        center_part=all_slices[int(nb_slices/2)]
        k_z_new= np.zeros((timesteps, nb_rep))
        for j in range( k_z.shape[0]):
            num_center_part=np.argwhere(k_z[j]==center_part)[0][0]
            #print(num_center_part)
            k_z_new[j,:num_center_part]=k_z[j,:num_center_part]
            k_z_new[j,(num_center_part+nb_rep_center_part):]=k_z[j,(num_center_part+1):]
        print(k_z_new[0,:])
        k_z=k_z_new


    k_z=np.repeat(k_z, nspoke, axis=0)

    print(k_z.shape)
    k_z = np.expand_dims(k_z, axis=-1)


    traj = np.expand_dims(all_spokes, axis=-2)

    print(traj.shape)
    k_z, traj = np.broadcast_arrays(k_z, traj)

    # k_z = np.reshape(k_z, (timesteps, -1))
    # traj = np.reshape(traj, (timesteps, -1))

    result = np.stack([traj.real,traj.imag, k_z], axis=-1)
    print(result.shape)
    return result.reshape(result.shape[0],-1,result.shape[-1])


def distrib_angle_traj_3D(total_nspoke, npoint, nspoke, nb_slices, undersampling_factor=4):
    timesteps = int(total_nspoke / nspoke)
    nb_rep = int(nb_slices / undersampling_factor)
    all_spokes = distrib_angle_traj(total_nspoke, npoint)
    #traj = np.reshape(all_spokes, (-1, nspoke * npoint))

    k_z = np.zeros((timesteps, nb_rep))
    #all_slices = np.linspace(-np.pi, np.pi, nb_slices)
    all_slices=np.arange(-np.pi, np.pi, 2 * np.pi / nb_slices)
    k_z[0, :] = all_slices[::undersampling_factor]

    for j in range(1, k_z.shape[0]):
        k_z[j, :] = np.sort(np.roll(all_slices, -j)[::undersampling_factor])

    k_z=np.repeat(k_z, nspoke, axis=0)
    k_z = np.expand_dims(k_z, axis=-1)
    traj = np.expand_dims(all_spokes, axis=-2)
    k_z, traj = np.broadcast_arrays(k_z, traj)

    # k_z = np.reshape(k_z, (timesteps, -1))
    # traj = np.reshape(traj, (timesteps, -1))

    result = np.stack([traj.real,traj.imag, k_z], axis=-1)
    return result.reshape(result.shape[0],-1,result.shape[-1])

def spherical_golden_angle_means_traj_3D(total_nspoke, npoint, npart, undersampling_factor=4,k_max=np.pi):
    
    phi1=0.46557123
    phi2=0.6823278
    # phi1=0.4656
    # phi2=0.6823

    theta=2*np.pi*np.mod(np.arange(total_nspoke*npart)*phi2,1)
    phi=np.arccos(np.mod(np.arange(total_nspoke*npart)*phi1,1))

    rotation=np.stack([np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)],axis=1).reshape(-1,1,3)
    base_spoke = (-k_max+k_max/(npoint)+np.arange(npoint)*2*k_max/(npoint))

    base_spoke=base_spoke.reshape(-1,1)
    spokes=np.matmul(base_spoke,rotation).reshape(npart,total_nspoke,npoint,-1)
    spokes=np.moveaxis(spokes,0,1)
    return spokes.reshape(total_nspoke,-1,3)




# def radial_golden_angle_traj_3D_incoherent(total_nspoke, npoint, nspoke, nb_slices, undersampling_factor=4,mode="old",offset=0):
#     timesteps = int(total_nspoke / nspoke)
#     nb_rep = int(nb_slices / undersampling_factor)
#     all_spokes = radial_golden_angle_traj(total_nspoke, npoint)
#     golden_angle = 111.246 * np.pi / 180
#     if mode=="old":
#         all_rotations = np.exp(1j * np.arange(nb_rep) * total_nspoke * golden_angle)
#     elif mode=="new":
#         all_rotations = np.exp(1j * np.arange(nb_rep) * golden_angle)
#     else:
#         raise ValueError("Unknown value for mode")
#     all_spokes = np.repeat(np.expand_dims(all_spokes, axis=1), nb_rep, axis=1)
#     traj = all_rotations[np.newaxis, :, np.newaxis] * all_spokes
#
#     k_z = np.zeros((timesteps, nb_rep))
#     all_slices = np.linspace(-np.pi, np.pi, nb_slices)
#     k_z[0, :] = all_slices[offset::undersampling_factor]
#     for j in range(1, k_z.shape[0]):
#         k_z[j, :] = np.sort(np.roll(all_slices, -j)[offset::undersampling_factor])
#
#     k_z=np.repeat(k_z, nspoke, axis=0)
#     k_z = np.expand_dims(k_z, axis=-1)
#     k_z, traj = np.broadcast_arrays(k_z, traj)
#
#     result = np.stack([traj.real,traj.imag, k_z], axis=-1)
#     return result.reshape(result.shape[0],-1,result.shape[-1])

def radial_golden_angle_traj_3D_incoherent(total_nspoke, npoint, nspoke, nb_slices, undersampling_factor=1,mode="old",offset=0):
    timesteps = int(total_nspoke / nspoke)
    nb_rep = math.ceil(nb_slices / undersampling_factor)

    print(nb_rep)
    print(nb_slices)
    print(undersampling_factor)

    golden_angle = 111.246 * np.pi / 180
    #all_slices = np.linspace(-np.pi, np.pi, nb_slices)
    all_slices = np.arange(-np.pi, np.pi, 2 * np.pi / nb_slices)

    # all_spokes = radial_golden_angle_traj(total_nspoke, npoint)
    # if mode=="old":
    #     all_rotations = np.exp(1j * np.arange(nb_rep) * total_nspoke * golden_angle)
    # elif mode=="new":
    #     all_rotations = np.exp(1j * np.arange(nb_rep) * golden_angle)
    # else:
    #     raise ValueError("Unknown value for mode")
    # all_spokes = np.repeat(np.expand_dims(all_spokes, axis=1), nb_rep, axis=1)
    # traj = all_rotations[np.newaxis, :, np.newaxis] * all_spokes
    #
    # k_z = np.zeros((timesteps, nb_rep))
    # k_z[0, :] = all_slices[offset::undersampling_factor]
    # for j in range(1, k_z.shape[0]):
    #     k_z[j, :] = np.sort(np.roll(all_slices, -j)[offset::undersampling_factor])
    #
    # k_z=np.repeat(k_z, nspoke, axis=0)
    # k_z = np.expand_dims(k_z, axis=-1)
    # k_z, traj = np.broadcast_arrays(k_z, traj)
    #
    # result = np.stack([traj.real,traj.imag, k_z], axis=-1)
    # return result.reshape(result.shape[0],-1,result.shape[-1])

    all_spokes = radial_golden_angle_traj(total_nspoke, npoint)
    if mode=="old":
        all_rotations = np.exp(1j * np.arange(nb_slices) * total_nspoke * golden_angle)
    elif mode=="new":
        all_rotations = np.exp(1j * np.arange(nb_slices) * golden_angle)
    else:
        raise ValueError("Unknown value for mode")

    all_spokes = np.repeat(np.expand_dims(all_spokes, axis=1), nb_slices, axis=1)
    traj = all_rotations[np.newaxis, :, np.newaxis] * all_spokes

    k_z=np.zeros((timesteps, nb_slices))
    k_z[0, :] = all_slices
    for j in range(1, k_z.shape[0]):
        k_z[j, :] = np.sort(np.roll(all_slices, -j))

    print(traj.shape)
    k_z=np.repeat(k_z, nspoke, axis=0)
    k_z = np.expand_dims(k_z, axis=-1)
    k_z, traj = np.broadcast_arrays(k_z, traj)

    result = np.stack([traj.real,traj.imag, k_z], axis=-1)

    print(result.shape)

    if undersampling_factor>1:
        print(result.shape)
        result = result.reshape(timesteps, nspoke, -1, npoint, result.shape[-1])

        result_us=np.zeros((timesteps, nspoke, nb_rep, npoint, 3),
                          dtype=result.dtype)
        
        #result_us[:, :, :, :, 1:] = result[:, :, :nb_rep, :, 1:]
        #print(result_us.shape)
        shift = offset

        for sl in range(nb_slices):

            if int(sl/undersampling_factor)<nb_rep:
                result_us[shift::undersampling_factor, :, int(sl/undersampling_factor), :, :] = result[shift::undersampling_factor, :, sl, :, :]
                shift += 1
                shift = shift % (undersampling_factor)
            else:
                continue

        result=result_us


    return result.reshape(total_nspoke,-1,3)


def radial_golden_angle_traj_random_3D(total_nspoke, npoint, nspoke, nb_slices, undersampling_factor=4,frac_center=0.25,mode="old",incoherent=True):
    timesteps = int(total_nspoke / nspoke)
    nb_rep = int(nb_slices / undersampling_factor)
    all_spokes = radial_golden_angle_traj(total_nspoke, npoint)
        #traj = np.reshape(all_spokes, (-1, nspoke * npoint))

    if incoherent:
        golden_angle = 111.246 * np.pi / 180
        if mode=="old":
            all_rotations = np.exp(1j * np.arange(nb_rep) * total_nspoke * golden_angle)
        elif mode=="new":
            all_rotations = np.exp(1j * np.arange(nb_rep) * golden_angle)
        else:
            raise ValueError("Unknown value for mode")
        all_spokes = np.repeat(np.expand_dims(all_spokes, axis=1), nb_rep, axis=1)
        traj = all_rotations[np.newaxis, :, np.newaxis] * all_spokes
    else:
        traj = np.expand_dims(all_spokes, axis=-2)

    k_z = np.zeros((timesteps, nb_rep))
    all_slices = np.linspace(-np.pi, np.pi, nb_slices)
    kz_center=all_slices[(int(nb_slices/2)+np.array(range(int(-frac_center*nb_rep/2),int(frac_center*nb_rep/2),1)))]
    kz_border = [k for k in all_slices if k not in kz_center]
    nb_border=nb_rep-len(kz_center)
    for j in range(k_z.shape[0]):
        k_z[j, :] = np.sort(np.concatenate([np.random.choice(kz_border,size=int(nb_border),replace=False),kz_center]))

    k_z=np.repeat(k_z, nspoke, axis=0)
    k_z = np.expand_dims(k_z, axis=-1)

    k_z, traj = np.broadcast_arrays(k_z, traj)

    # k_z = np.reshape(k_z, (timesteps, -1))
    # traj = np.reshape(traj, (timesteps, -1))

    result = np.stack([traj.real,traj.imag, k_z], axis=-1)
    return result.reshape(result.shape[0],-1,result.shape[-1])

def spiral_golden_angle_traj(total_spiral,fov, N, f_sampling, R, ninterleaves, alpha, gm, sm):
    golden_angle = 111.246 * np.pi / 180
    base_spiral = spiral(fov, N, f_sampling, R, ninterleaves, alpha, gm, sm)
    base_spiral = base_spiral[:,0]+1j*base_spiral[:,1]
    all_rotations = np.exp(1j * np.arange(total_spiral) * golden_angle)
    all_spirals = np.matmul(np.diag(all_rotations), np.repeat(base_spiral.reshape(1, -1), total_spiral, axis=0))
    return all_spirals

def spiral_golden_angle_traj_v2(total_spiral,nspiral,fov, N, f_sampling, R, ninterleaves, alpha, gm, sm):
    golden_angle = 111.246 * np.pi / 180
    angle = 2*np.pi/nspiral
    base_spiral = spiral(fov, N, f_sampling, R, ninterleaves, alpha, gm, sm)
    base_spiral = base_spiral[:,0]+1j*base_spiral[:,1]
    all_rotations = np.exp(1j * np.arange(total_spiral) * angle)
    all_spirals = np.matmul(np.diag(all_rotations), np.repeat(base_spiral.reshape(1, -1), total_spiral, axis=0))

    disk_rotations = np.exp(1j * np.arange(int(total_spiral/nspiral)) * golden_angle)
    disk_rotations = np.repeat(disk_rotations,nspiral)
    all_spirals = np.matmul(np.diag(disk_rotations),all_spirals )

    return all_spirals


def groupby(arr, n, axis=0, mode="edge"):
    """ group array into groups of size 'n' """

    ngroup = -(-arr.shape[axis] // n)
    if arr.shape[axis] % n != 0:
        # pad array
        padding = [(0,0)] * arr.ndim
        nzero = n - np.mod(arr.shape[axis], n)
        padding[axis] = (nzero//2, -(-nzero//2))
        arr = np.pad(arr, padding, mode=mode)
    arr = np.moveaxis(arr, axis, 0)
    arr = arr.reshape((ngroup, -1) + arr.shape[1:])
    return list(np.moveaxis(arr, 1, axis + 1))