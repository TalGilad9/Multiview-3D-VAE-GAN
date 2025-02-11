import scipy.ndimage as nd
import scipy.io as io
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.measure as sk
from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec
import numpy as np
from torch.utils import data
from torch.autograd import Variable
import torch
import os
import pickle
from skimage.io import imread
import trimesh
from PIL import Image
from torchvision import transforms
import binvox_rw
import glob

def getVoxelFromMat(path, cube_len=64):
    """Mat 데이터로 부터 Voxel 을 가져오는 함수"""
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
    return voxels

def getVolumeFromBinvox(path):
    with open(path, 'rb') as file:
        data = np.int32(binvox_rw.read_as_3d_array(file).data)
    return data

def getVolumeFromOFF(path, sideLen=32):
    mesh = trimesh.load(path)
    volume = trimesh.voxel.VoxelMesh(mesh, 0.5).matrix
    (x, y, z) = map(float, volume.shape)
    volume = nd.zoom(volume.astype(float),
                     (sideLen/x, sideLen/y, sideLen/z),
                     order=1,
                     mode='nearest')
    volume[np.nonzero(volume)] = 1.0
    return volume.astype(np.bool)


def getVFByMarchingCubes(voxels, threshold=0.5):
    """Voxel 로 부터 Vertices, faces 리턴 하는 함수"""
    v, f = sk.marching_cubes_classic(voxels, level=threshold)
    return v, f


def plotVoxelVisdom(voxels, visdom, title):
    v, f = getVFByMarchingCubes(voxels)
    visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))


def plotFromVoxels(voxels):
    z, x, y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='red')
    plt.savefig('test')
    plt.show()


def SavePloat_Voxels(voxels, path, iteration):
    voxels = voxels[:8].__ge__(0.5)
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
    #plt.show()
    plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()

    with open(path + '/{}.pkl'.format(str(iteration).zfill(3)), "wb") as f:
        pickle.dump(voxels, f, protocol=pickle.HIGHEST_PROTOCOL)


def make_hyparam_string(hyparam_dict):
    str_result = ""
    for i in hyparam_dict.keys():
        str_result = str_result + str(i) + "=" + str(hyparam_dict[i]) + "_"
    return str_result[:-1]

class ShapeNetDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader"""

    def __init__(self, root, args):
        """Set the path for Data.

        Args:
            root: image directory.
            transform: Tensor transformer.
        """
        self.root = root
        self.listdir = os.listdir(self.root)
        self.args = args

    def __getitem__(self, index):
        #print("images=")
        #print(images.size())
        #with open(self.root + self.listdir[index], "rb") as f:
            #volume = np.asarray(getVoxelFromMat(f, self.args.cube_len), dtype=np.float32)
            #plotFromVoxels(volume)
        model_3d_file = [name for name in self.listdir if name.endswith('.' + "binvox")][index]
        volume = np.asarray(getVolumeFromBinvox(self.root + model_3d_file), dtype=np.float32)
        return torch.FloatTensor(volume)

    def __len__(self):
        return len([name for name in self.listdir if name.endswith('.' + "binvox")])

class ShapeNetPlusImageDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader"""

    def __init__(self, root, args):
        """Set the path for Data.

        Args:
            root: image directory.
            transform: Tensor transformer.
        """
        self.root = root
        self.listdir = os.listdir(self.root)
        self.args = args
        self.img_size = args.image_size
        self.p = transforms.Compose([transforms.Scale((self.img_size, self.img_size))])

    def __getitem__(self, index):

        model_3d_file = [name for name in self.listdir if name.endswith('.' + "binvox")][index]

        model_2d_file = model_3d_file[:-7] + "_002.png"
        #with open(self.root + model_3d_file, "rb") as f:
        volume = np.asarray(getVolumeFromBinvox(self.root + model_3d_file), dtype=np.float32)
        #print(volume.shape)
        #plotFromVoxels(volume)
        #with open(self.root + model_2d_file, "rb") as g:
        #image = np.array(imread(self.root + model_2d_file))
        image = Image.open(self.root + model_2d_file)
        image = np.asarray(self.p(image))
        return (torch.FloatTensor(image), torch.FloatTensor(volume) )

    def __len__(self):
        return len( [name for name in self.listdir if name.endswith('.' + "binvox")])

class ShapeNetMultiviewDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader"""

    def __init__(self, root, args):
        """Set the path for Data.

        Args:
            root: image directory.
            transform: Tensor transformer.
        """
        print("inittttT")
        self.root = root
        self.listdir = os.listdir(self.root)
        self.args = args
        self.img_size = args.image_size
        self.p = transforms.Compose([transforms.Scale((self.img_size, self.img_size))])

    def __getitem__(self, index):
        model_3d_file = [name for name in self.listdir if name.endswith('.' + "binvox")][index]

        model_2d_files = [name for name in self.listdir if name.startswith(model_3d_file[:-7]) and name.endswith(".png")][:3]
        #with open(self.root + model_3d_file, "rb") as f:
        volume = np.asarray(getVolumeFromBinvox(self.root + model_3d_file), dtype=np.float32)
        #print(volume.shape)
        #plotFromVoxels(volume)
        #with open(self.root + model_2d_file, "rb") as g:
        #image = np.array(imread(self.root + model_2d_file))
        images = [torch.FloatTensor(np.asarray(self.p(Image.open(self.root +x )))) for x in model_2d_files]

        return (images, torch.FloatTensor(volume))

    def __len__(self):
        return len( [name for name in self.listdir if name.endswith('.' + "binvox")])


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def generateZ(args, batch_size=-1):
    if (batch_size == -1):
      if args.z_dis == "norm":
          Z = var_or_cuda(torch.Tensor(args.batch_size, args.z_size).normal_(0, 0.33))
      elif args.z_dis == "uni":
          Z = var_or_cuda(torch.randn(args.batch_size, args.z_size))
      else:
          print("z_dist is not normal or uniform")
    else:
      if args.z_dis == "norm":
          Z = var_or_cuda(torch.Tensor(batch_size, args.z_size).normal_(0, 0.33))
      elif args.z_dis == "uni":
          Z = var_or_cuda(torch.randn(batch_size, args.z_size))
      else:
          print("z_dist is not normal or uniform")
    
    return Z

########################## Pickle helper ###############################


def read_pickle(path, G, G_solver, D_, D_solver,E_=None,E_solver = None ):
    try:

        files = os.listdir(path)
        file_list = [int(file.split('_')[-1].split('.')[0]) for file in files]
        file_list.sort()
        recent_iter = str(file_list[-1])
        print(recent_iter, path)

        with open(path + "/G_" + recent_iter + ".pkl", "rb") as f:
            G.load_state_dict(torch.load(f))
        with open(path + "/G_optim_" + recent_iter + ".pkl", "rb") as f:
            G_solver.load_state_dict(torch.load(f))
        with open(path + "/D_" + recent_iter + ".pkl", "rb") as f:
            D_.load_state_dict(torch.load(f))
        with open(path + "/D_optim_" + recent_iter + ".pkl", "rb") as f:
            D_solver.load_state_dict(torch.load(f))
        if E_ is not None:
            with open(path + "/E_" + recent_iter + ".pkl", "rb") as f:
                E_.load_state_dict(torch.load(f))
            with open(path + "/E_optim_" + recent_iter + ".pkl", "rb") as f:
                E_solver.load_state_dict(torch.load(f))


    except Exception as e:

        print("fail try read_pickle", e)



def save_new_pickle(path, iteration, G, G_solver, D_, D_solver, E_=None,E_solver = None):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + "/G_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(G.state_dict(), f)
    with open(path + "/G_optim_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(G_solver.state_dict(), f)
    with open(path + "/D_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(D_.state_dict(), f)
    with open(path + "/D_optim_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(D_solver.state_dict(), f)
    if E_ is not None:
        with open(path + "/E_" + str(iteration) + ".pkl", "wb") as f:
            torch.save(E_.state_dict(), f)
        with open(path + "/E_optim_" + str(iteration) + ".pkl", "wb") as f:
            torch.save(E_solver.state_dict(), f)