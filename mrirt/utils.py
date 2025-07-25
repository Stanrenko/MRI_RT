import sys
import numpy as np
import os
os.environ['PATH'] = os.environ['TOOLBOX_PATH'] + ":" + os.environ['PATH']
sys.path.append(os.environ['TOOLBOX_PATH'] + "/python/")

import twixtools
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt


from .trajectory import Radial
from bart import bart
import cfl
import json
import pywt

from pathlib import Path




def plot_image_grid(list_images,nb_row_col,figsize=(10,10),title="",cmap=None,save_file=None,same_range=False,aspect=None):
    """
    Plot a grid of images using matplotlib.

    Args:
        list_images (list): List of images to plot.
        nb_row_col (tuple): Number of rows and columns in the grid.
        figsize (tuple): Size of the figure.
        title (str): Title of the plot.
        cmap: Colormap for images.
        save_file (str): Path to save the figure.
        same_range (bool): Use same color range for all images.
        aspect: Aspect ratio for images.
    """
    fig = plt.figure(figsize=figsize)
    plt.title(title)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=nb_row_col,  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    if same_range:
        vmin=np.min(np.array(list_images))
        vmax=np.max(np.array(list_images))
        for ax, im in zip(grid, list_images):
            ax.imshow(im,cmap=cmap,vmin=vmin,vmax=vmax,aspect=aspect)
    else:
        for ax, im in zip(grid, list_images):
            ax.imshow(im,cmap=cmap,aspect=aspect)

    if save_file is not None:
        plt.savefig(save_file)
        plt.show()
    else:
        plt.show()


def generate_gif(volumes_all,filename_gif="motion.gif"):
    """
    Plot a grid of images using matplotlib.

    Args:
        list_images (list): List of images to plot.
        nb_row_col (tuple): Number of rows and columns in the grid.
        figsize (tuple): Size of the figure.
        title (str): Title of the plot.
        cmap: Colormap for images.
        save_file (str): Path to save the figure.
        same_range (bool): Use same color range for all images.
        aspect: Aspect ratio for images.
    """

    gif = []
    images_gif=np.abs(volumes_all)
    for i in range(volumes_all.shape[0]):
        img = Image.fromarray(np.uint8(images_gif[i] / np.max(images_gif[i]) * 255), 'L')
        img = img.convert("P")
        gif.append(img)
    gif[0].save(filename_gif, save_all=True, append_images=gif[1:], duration=20, loop=0)


def read_data(filename,index=-1):
    """
    Read raw Siemens MRI data and generate BART-compatible k-space and trajectory files.

    Args:
        filename (str or Path): Path to raw data file.
        index (int): Index for twix file selection.
    """
    filename=Path(filename)
    print("Reading raw data")
    twix = twixtools.read_twix(str(filename))
    mdb_list=twix[index]["mdb"]

    data_bart_raw = []


    for i, mdb in enumerate(mdb_list):
        if mdb.is_image_scan():
            data_bart_raw.append(mdb)

    data_bart_raw = np.array([mdb.data for mdb in data_bart_raw])
    data_bart_raw=np.moveaxis(data_bart_raw,0,1)
    print(data_bart_raw.shape)

    ncoils,nspoke,npoint=data_bart_raw.shape
    print(data_bart_raw.shape)

    data_bart_raw=np.moveaxis(data_bart_raw,(0,1,2),(2,1,0))
    data_bart_raw=data_bart_raw[np.newaxis,:,:,:]
    
    data_bart_raw=(data_bart_raw)/np.max(np.abs(data_bart_raw))*2048
    
    cfl.writecfl(str(filename.parent / "data_bart_raw"), data_bart_raw)



    print("Generating trajectory data")
    radial_traj=Radial(total_nspokes=nspoke,npoint=npoint)
    traj_python = radial_traj.get_traj()
    traj_python=traj_python.reshape(nspoke,-1,2)
    traj_python=np.moveaxis(traj_python,(0,1,2),(2,1,0))
    traj_python[0] = traj_python[0] / np.max(np.abs(traj_python[0])) * (int(
            npoint / 2)-0.5)
    traj_python[1] = traj_python[1] / np.max(np.abs(traj_python[1])) * (int(
                npoint / 2)-0.5)

    traj_python_bart=traj_python.reshape(2,npoint,-1)
    traj_python_bart=np.pad(traj_python_bart,((0,1), (0, 0), (0, 0)), mode='constant')
    print(traj_python_bart.shape)
    cfl.writecfl(str(filename.parent / "traj"), traj_python_bart)
    
    print("Writing sequence parameters")
    paramseq={"ncoils":ncoils,
              "nspokes": nspoke, 
              "npoint": npoint}
    
    with open(str(filename.parent / "param_seq.json"), 'w') as fp:
        json.dump(paramseq, fp)




def coil_compression_bart(data_bart, traj_bart,dirname=".",ncoils=8,lowmem=False, inverse_nufft=False,calc_sensi=False,excluded_channels=None):
    """
    Perform coil compression using BART and save results.

    Args:
        data_bart (np.ndarray): K-space data.
        traj_bart (np.ndarray): Trajectory data.
        dirname (str): Output directory.
        ncoils (int): Number of virtual coils.
        lowmem (bool): Use low memory mode.
        inverse_nufft (bool): Use inverse NUFFT.
        calc_sensi (bool): Calculate coil sensitivities.
        excluded_channels (list): Channels to exclude.
    """
    dirname=Path(dirname)
    if lowmem:
        suffix_lowmem="--lowmem --no-precomp"
    else:
        suffix_lowmem=""

    if excluded_channels is not None:
        channels=set(range(data_bart.shape[3]))
        kept_channels=np.array(list(channels-set(excluded_channels)))
        print("Kept channels: {}".format(kept_channels))
        data_bart=data_bart[:,:,:,kept_channels]

    print("Coil images")
    if inverse_nufft:
        print("Using inverse nufft - might take a few minutes")
        coil_img = bart(1,'nufft {} -i -t'.format(suffix_lowmem), traj_bart, data_bart)
    else:
        print("Using adjoint with density adjustment to approximate inverse nufft")
        npoint=data_bart.shape[1]
        density = np.abs(np.linspace(-1, 1, npoint))
        density=density[np.newaxis,:,np.newaxis,np.newaxis]
        coil_img = bart(1,'nufft {} -a -t'.format(suffix_lowmem), traj_bart, data_bart*density)

    coil_img_plot=np.moveaxis(coil_img,-2,0)
    coil_img_plot=np.moveaxis(coil_img_plot,-1,0)
    sl = int(coil_img_plot.shape[1]/2)
    list_images=list(np.abs(coil_img_plot[:,sl,:,:]))
    plot_image_grid(list_images,(6,6),title="BART Coil Images for slice".format(sl),save_file=str(dirname / "coil_img.jpeg"))

    print("Regridding to cartesian grid")
    data_cart = bart(1,'fft -u 7', coil_img)
    cc=bart(1,"cc -M",data_cart)
    cfl.writecfl(str(dirname / "cc"),cc)

    print("Applying coil compression to k-space data")
    data_bart_cc = bart(1, 'ccapply -p {}'.format(ncoils), data_bart, cc)
    cfl.writecfl(str(dirname / "data_bart_cc"), data_bart_cc)
    # kdata_python_cc=kdata_bart_cc.squeeze().reshape(npoint,nb_segments,nb_part,n_comp)
    # kdata_python_cc=np.moveaxis(kdata_python_cc,-1,0)
    # kdata_python_cc=np.moveaxis(kdata_python_cc,1,-1)

    if calc_sensi:
        print("Calculating coil sensi")
        data_cart_cc = bart(1, 'ccapply -p {}'.format(ncoils), data_cart, cc)
        b1_bart_cc=bart(1,"ecalib -a -m1",data_cart_cc)
        # b1_bart_cc=bart(1,"ecalib -a -m1",data_cart)
        
        cfl.writecfl(str(dirname / "b1_bart_cc"), b1_bart_cc)

        b1_python_cc=np.moveaxis(b1_bart_cc,-2,0)
        b1_python_cc=np.moveaxis(b1_python_cc,-1,0)

        sl = int(b1_python_cc.shape[1]/2)

        list_images=list(np.abs(b1_python_cc[:,sl,:,:]))
        plot_image_grid(list_images,(6,6),title="BART Coil Sensitivity map for slice".format(sl),save_file=str(dirname / "b1_img.jpeg"))


def reco_rtnlinv(data_bart_rt,traj_bart_rt,dirname=".",param_reco=None,niter=3,framebounds=None):
    """
    Reconstruct images using BART RTNLINV algorithm.

    Args:
        data_bart_rt (np.ndarray): K-space data.
        traj_bart_rt (np.ndarray): Trajectory data.
        dirname (str): Output directory.
        param_reco (dict): Reconstruction parameters.
        niter (int): Number of iterations.
        framebounds (tuple): Frame bounds for reconstruction.
    """
    if param_reco is None:
        suffix=""
    if framebounds is not None:
        traj_bart_rt=traj_bart_rt[:,:,:,:,:,:,:,:,:,:,framebounds[0]:framebounds[1]]
        data_bart_rt=data_bart_rt[:,:,:,:,:,:,:,:,:,:,framebounds[0]:framebounds[1]]
           
    print("Reconstruction of the serie of images with RTNLINV")
    img=bart(1,"rtnlinv -w1. -d5 -i{}{} -t".format(niter,suffix),traj_bart_rt,data_bart_rt)
    cfl.writecfl(str(dirname / "img_bart_rt"), img)

def reco_pics(data_bart_rt,traj_bart_rt,b1,dirname=".",param_reco=None,niter=3,framebounds=None,use_mt=True,nproc=None):
    """
    Reconstruct images using BART PICS algorithm.

    Args:
        data_bart_rt (np.ndarray): K-space data.
        traj_bart_rt (np.ndarray): Trajectory data.
        b1 (np.ndarray): Coil sensitivity maps.
        dirname (str): Output directory.
        param_reco (dict): Reconstruction parameters.
        niter (int): Number of iterations.
        framebounds (tuple): Frame bounds for reconstruction.
        use_mt (bool): Use multi-threading.
        nproc (int): Number of threads.
    """

    suffix=""
    if framebounds is not None:
        traj_bart_rt=traj_bart_rt[:,:,:,:,:,:,:,:,:,:,framebounds[0]:framebounds[1]]
        data_bart_rt=data_bart_rt[:,:,:,:,:,:,:,:,:,:,framebounds[0]:framebounds[1]]
    
    if "use_infimal_conv" in param_reco:
        use_infimal_conv=param_reco["use_infimal_conv"]
    else:
        use_infimal_conv=False

    if "lambda_LLR" in param_reco:
        suffix+=" -RL:7:7:{}".format(param_reco["lambda_LLR"])
        if "block_size" in param_reco:
            suffix+=" -N -b {}".format(param_reco["block_size"])
        else:
            suffix+=" -N -b 8"
    if "lambda_wav" in param_reco:
        suffix+=" -RW:7:0:{}".format(param_reco["lambda_wav"])

    if "lambda_TVt" in param_reco and not use_infimal_conv:
        if "dims_TV" in param_reco:
            sorted_dims=np.sort(np.array(param_reco["dims_TV"]))
            powers= 2**sorted_dims
            tv_dims = np.sum(powers)
        else:
            tv_dims=1024
        suffix+=" -RT:{}:0:{}".format(tv_dims,param_reco["lambda_TVt"])
        if "max_iter_cg" in param_reco:
            suffix+=" -C {}".format(param_reco["max_iter_cg"])

    if "lambda_TVx" in param_reco and not use_infimal_conv:
        if "dims_TVx" in param_reco:
            sorted_dims=np.sort(np.array(param_reco["dims_TVx"]))
            powers= 2**sorted_dims
            tvx_dims = np.sum(powers)
        else:
            tvx_dims=7
        suffix+=" -RT:{}:0:{}".format(tvx_dims,param_reco["lambda_TVx"])
        if "max_iter_cg" in param_reco:
            suffix+=" -C {}".format(param_reco["max_iter_cg"])

    if "lambda_TGVt" in param_reco:
        suffix+=" -RG:1024:0:{}".format(param_reco["lambda_TGVt"])
        if "max_iter_cg" in param_reco:
            suffix+=" -C {}".format(param_reco["max_iter_cg"])
    

    if use_infimal_conv:
        if "dims_TV" in param_reco:
            sorted_dims=np.sort(np.array(param_reco["dims_TV"]))
            powers= 2**sorted_dims
            tv_dims = np.sum(powers)
        else:
            tv_dims=1024

        if "dims_TVx" in param_reco:
            sorted_dims=np.sort(np.array(param_reco["dims_TVx"]))
            powers= 2**sorted_dims
            tvx_dims = np.sum(powers)
        else:
            tvx_dims=7
        suffix+=" -RC:{}:0:{} -RC:{}:0:{}".format(tv_dims,param_reco["lambda_TVt"],tvx_dims,param_reco["lambda_TVx"])
    
    
    bart_command="pics -d5 -e -S -i{}{} -t".format(niter,suffix)
    if use_mt:
        if nproc is None:
            nproc = os.cpu_count()
        print("Using {} threads for PICS".format(nproc))
        os.environ['OMP_NUM_THREADS']= str(nproc)

    print("BART COMMMAND : {} ".format(bart_command))
    print("Reconstruction of the serie of images with PICS")
    img=bart(1,bart_command,traj_bart_rt,data_bart_rt,b1)
    cfl.writecfl(str(dirname / "img_bart_rt_pics"), img)



def estimate_sigma_wavelet(img):
    """
    Estimate noise sigma using wavelet high-frequency coefficients.

    Args:
        img (np.ndarray): 2D image array.

    Returns:
        float: Estimated noise sigma.
    """
    coeffs = pywt.wavedec2(img, 'db1', level=1)
    # Use the detail coefficients (high-freq components)
    detail_coeffs = coeffs[-1]  # (cH, cV, cD)
    all_details = np.concatenate([d.ravel() for d in detail_coeffs])
    sigma_est = np.median(np.abs(all_details)) / 0.6745
    return sigma_est


def estimate_sigma_wavelet_3d(volume):
    """
    Estimate noise sigma from a 3D volume using wavelet detail coefficients.

    Args:
        volume (np.ndarray): 3D image volume.

    Returns:
        float: Estimated noise sigma.
    """
    coeffs = pywt.dwtn(volume, wavelet='db1', axes=(0,1,2))
    # Get all high-frequency subbands
    detail_coeffs = [v.ravel() for k, v in coeffs.items() if k != 'aaa']
    all_details = np.concatenate(detail_coeffs)
    # print(all_details)
    sigma_est = np.median(np.abs(all_details)) / 0.6745
    return sigma_est



#TODO - adapt 3D code below

# @ma.machine()
# @ma.parameter("filename_kdata", str, default=None, description="MRF raw data")
# @ma.parameter("filename_seqParams", str, default=None, description="Seq params")
# @ma.parameter("dens_adj", bool, default=False, description="Radial density adjustment")
# @ma.parameter("lowmem", bool, default=False, description="Low memory nufft in bart")
# @ma.parameter("n_comp", int, default=None, description="Number of virtual coils")
# @ma.parameter("filename_cc", str, default=None, description="Filename for coil compression")
# @ma.parameter("calc_sensi", bool, default=True, description="Calculate coil sensitivities")
# @ma.parameter("iskushball", bool, default=False, description="3D Kushball sampling")
# @ma.parameter("spoke_start", int, default=None, description="Starting segment to avoid inversion")
# @ma.parameter("us", int, default=1, description="undersampling partitions")

# def coil_compression_bart(filename_kdata,dens_adj,n_comp,filename_cc,calc_sensi,iskushball,filename_seqParams,spoke_start,us,lowmem):
#     """
#     Perform coil compression for 3D MRI data using BART.

#     Args:
#         filename_kdata (str): Path to k-space data file.
#         dens_adj (bool): Apply density adjustment.
#         n_comp (int): Number of virtual coils.
#         filename_cc (str): Path to coil compression file.
#         calc_sensi (bool): Calculate coil sensitivities.
#         iskushball (bool): Use Kushball sampling.
#         filename_seqParams (str): Path to sequence parameters file.
#         spoke_start (int): Starting segment index.
#         us (int): Undersampling factor.
#         lowmem (bool): Use low memory mode.
#     """
#     kdata_all_channels_all_slices = np.load(filename_kdata)

#     nb_channels,nb_segments,nb_part,npoint=kdata_all_channels_all_slices.shape

#     print(kdata_all_channels_all_slices.shape)


#     filename_virtualcoils = str.split(filename_kdata, "_kdata.npy")[0] + "_bart{}_virtualcoils_{}.pkl".format(n_comp,n_comp)
#     filename_b12Dplus1 = str.split(filename_kdata, "_kdata.npy")[0] + "_bart{}_b12Dplus1_{}.npy".format(n_comp,n_comp)
#     filename_coilimg = str.split(filename_kdata, "_kdata.npy")[0] + "_bart{}_coil_img_{}.npy".format(n_comp,n_comp)
#     filename_kdata_compressed = str.split(filename_kdata, "_kdata.npy")[0] + "_bart{}_kdata.npy".format(n_comp)
#     coil_image_file=str.split(filename_b12Dplus1, ".npy")[0] + "_coilimg.jpg"

#     if filename_seqParams is None:
#         filename_seqParams =filename_kdata.split("_kdata.npy")[0] + "_seqParams.pkl"

#     file = open(filename_seqParams, "rb")
#     dico_seqParams = pickle.load(file)
#     file.close()

#     print(dico_seqParams)

#     if "Spherical" in dico_seqParams.keys():
#         iskushball=dico_seqParams["Spherical"]
#     else:
#         iskushball=False

#     nb_slices = int(dico_seqParams["nb_part"])
    
#     if dens_adj:
#         print("Performing Density Adjustment....")
#         density = np.abs(np.linspace(-1, 1, npoint))
#         density = np.expand_dims(density, tuple(range(kdata_all_channels_all_slices.ndim - 1)))
#         if iskushball:
#             # phi1 = 0.4656
#             # phi = np.arccos(np.mod(np.arange(nb_segments * nb_slices) * phi1, 1))
#             kdata_all_channels_all_slices *= density**2#*np.sin(phi.reshape(nb_slices, nb_segments).T[None, :, :, None])
#         else:
#             kdata_all_channels_all_slices *= density
    
#     kdata_bart=np.moveaxis(kdata_all_channels_all_slices,-1,1)
#     kdata_bart=np.moveaxis(kdata_bart,0,-1)
#     kdata_bart=kdata_bart[None,:]

#     kdata_bart=kdata_bart.reshape(1,npoint,-1,nb_channels)

#     if lowmem:
#         suffix_lowmem="--lowmem --no-precomp"
#     else:
#         suffix_lowmem=""
    
#     if (filename_cc is None) or calc_sensi:
        

#         incoherent=False
#         if iskushball:
#             print("Kushball reco")
#             radial_traj=Radial3D(total_nspokes=nb_segments,undersampling_factor=1,npoint=npoint,nb_slices=nb_part,mode="Kushball")
#         else:
#             radial_traj = Radial3D(total_nspokes=nb_segments, undersampling_factor=1, npoint=npoint,
#                                 nb_slices=nb_part, incoherent=incoherent, mode=None,nspoke_per_z_encoding=nb_segments,)

#         traj_python = radial_traj.get_traj()
#         traj_python=traj_python.reshape(nb_segments,nb_part,-1,3)
#         if spoke_start is not None:
#             traj_python=traj_python[spoke_start:]
        
#         traj_python=traj_python.T
#         traj_python=np.moveaxis(traj_python,-1,-2)

#         traj_python[0] = traj_python[0] / np.max(np.abs(traj_python[0])) * int(
#         npoint / 4)
#         traj_python[1] = traj_python[1] / np.max(np.abs(traj_python[1])) * int(
#             npoint / 4)
#         if iskushball:
#             traj_python[2] = traj_python[2] / np.max(np.abs(traj_python[2])) * int(
#             npoint / 4)
        
#         else:
#             traj_python[2] = traj_python[2] / np.max(np.abs(traj_python[2])) * int(
#             nb_slices / 2)
        
#         print(traj_python.shape)

#         traj_python=traj_python[:,:,:,::us]
#         traj_python_bart=traj_python.reshape(3,npoint,-1)

#         print(traj_python_bart.shape)
#         print(kdata_bart.shape)

#         if spoke_start is not None:
#                 coil_img=bart(1,'nufft {} -a -t'.format(suffix_lowmem), traj_python_bart, (kdata_bart.reshape(1,npoint,nb_segments,nb_part,nb_channels)[:,:,spoke_start:,::us]).reshape(1,npoint,-1,nb_channels))
#         else:
#             coil_img = bart(1,'nufft {} -a -t'.format(suffix_lowmem), traj_python_bart, kdata_bart)

#         print(coil_img.shape)
        
#         coil_img_plot=np.moveaxis(coil_img,-2,0)
#         coil_img_plot=np.moveaxis(coil_img_plot,-1,0)
#         np.save(filename_coilimg,coil_img_plot)

#         sl = int(coil_img_plot.shape[1]/2)

#         list_images=list(np.abs(coil_img_plot[:,sl,:,:]))
#         plot_image_grid(list_images,(6,6),title="BART Coil Images for slice".format(sl),save_file=coil_image_file)
        
#         kdata_cart = bart(1,'fft -u 7', coil_img)

#     if filename_cc is None:
#         print("Calculating coil compression")
#         filename_cc = str.split(filename_kdata, "_kdata.npy")[0] + "_bart_cc.cfl"
#         cc=bart(1,"cc -M",kdata_cart)
#         cfl.writecfl(filename_cc,cc)
#     else:
#         print("Loading Coil compression")
#         cc=cfl.readcfl(filename_cc)

#     print("Applying coil compression to k-space data")
#     kdata_bart_cc = bart(1, 'ccapply -p {}'.format(n_comp), kdata_bart, cc)
#     kdata_python_cc=kdata_bart_cc.squeeze().reshape(npoint,nb_segments,nb_part,n_comp)
#     kdata_python_cc=np.moveaxis(kdata_python_cc,-1,0)
#     kdata_python_cc=np.moveaxis(kdata_python_cc,1,-1)
#     np.save(filename_kdata_compressed,kdata_python_cc)

#     if calc_sensi:
#         print("Calculating coil sensi")
#         kdata_cart_cc = bart(1, 'ccapply -p {}'.format(n_comp), kdata_cart, cc)
#         b1_bart_cc=bart(1,"ecalib -m1",kdata_cart_cc)
#         b1_python_cc=np.moveaxis(b1_bart_cc,-2,0)
#         b1_python_cc=np.moveaxis(b1_python_cc,-1,0)
#         np.save(filename_b12Dplus1,b1_python_cc)

#         image_file=str.split(filename_b12Dplus1, ".npy")[0] + ".jpg"

#         sl = int(b1_python_cc.shape[1]/2)

#         list_images=list(np.abs(b1_python_cc[:,sl,:,:]))
#         plot_image_grid(list_images,(6,6),title="BART Coil Sensitivity map for slice".format(sl),save_file=image_file)

#         pca_dict={}
#         for sl in range(nb_part):
#             pca=PCAComplex(n_components_=n_comp)
#             pca.explained_variance_ratio_=[1]
#             pca.components_=np.eye(n_comp)
#             pca_dict[sl]=deepcopy(pca)

#         with open(filename_virtualcoils, "wb") as file:
#             pickle.dump(pca_dict, file)

