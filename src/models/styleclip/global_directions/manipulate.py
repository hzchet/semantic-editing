import copy
from copy import deepcopy

import numpy as np
import torch

import src.models.styleclip.global_directions.dnnlib as dnnlib
import src.models.styleclip.global_directions.legacy as legacy

import types
from .training.networks import (
    SynthesisNetwork,
    SynthesisBlock,
    SynthesisLayer,
    ToRGBLayer,
)


def change_style_code(codes, layer, channel, step):
    codes[layer][:, channel] += step
    return codes


def LoadModel(network_pkl, device):
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)["G_ema"].requires_grad_(False)
        G = G.to(device)  # type: ignore

    G.synthesis.forward = types.MethodType(SynthesisNetwork.forward, G.synthesis)
    G.synthesis.W2S = types.MethodType(SynthesisNetwork.W2S, G.synthesis)

    for res in G.synthesis.block_resolutions:
        block = getattr(G.synthesis, f"b{res}")
        # print(block)
        block.forward = types.MethodType(SynthesisBlock.forward, block)

        if res != 4:
            layer = block.conv0
            layer.forward = types.MethodType(SynthesisLayer.forward, layer)
            layer.name = "conv0_resolution_" + str(res)

        layer = block.conv1
        layer.forward = types.MethodType(SynthesisLayer.forward, layer)
        layer.name = "conv1_resolution_" + str(res)

        layer = block.torgb
        layer.forward = types.MethodType(ToRGBLayer.forward, layer)
        layer.name = "toRGB_resolution_" + str(res)

    return G


def S2List(encoded_styles):
    all_s = []
    for name in encoded_styles.keys():
        tmp = encoded_styles[name].detach().cpu().numpy()
        all_s.append(tmp)
    return all_s


class Manipulator:
    def __init__(self, device='cpu', dataset_name="ffhq"):

        self.alpha = [0]  # manipulation strength
        self.num_images = 10
        self.img_index = 0  # which image to start
        # self.viz_size=256
        self.manipulate_layers = None  # which layer to manipulate, list
        self.truncation_psi = 0.7
        self.truncation_cutoff = 8

        #        self.G=LoadModel(self.model_path,self.model_name)

        self.LoadModel = LoadModel
        self.S2List = S2List

        fmaps = [512, 512, 512, 512, 512, 256, 128, 64, 32]
        self.fmaps = np.repeat(fmaps, 3)
        self.device = device

    def GetSName(self):
        s_names = []
        for res in self.G.synthesis.block_resolutions:
            if res == 4:
                tmp = f"conv1_resolution_{res}"
                s_names.append(tmp)

                tmp = f"toRGB_resolution_{res}"
                s_names.append(tmp)
            else:
                tmp = f"conv0_resolution_{res}"
                s_names.append(tmp)

                tmp = f"conv1_resolution_{res}"
                s_names.append(tmp)

                tmp = f"toRGB_resolution_{res}"
                s_names.append(tmp)

        return s_names

    def SL2D(self, tmp_code):
        encoded_styles = {}
        for i in range(len(self.s_names)):
            encoded_styles[self.s_names[i]] = torch.from_numpy(tmp_code[i]).to(
                self.device
            )

        return encoded_styles

    def GenerateS(self, num_img=100):
        seed = 5
        with torch.no_grad():
            z = torch.from_numpy(
                np.random.RandomState(seed).randn(num_img, self.G.z_dim)
            ).to(self.device)
            ws = self.G.mapping(
                z=z,
                c=None,
                truncation_psi=self.truncation_psi,
                truncation_cutoff=self.truncation_cutoff,
            )
            encoded_styles = self.G.synthesis.W2S(ws)
        #            encoded_styles=encoded_styles.cpu().numpy()

        self.dlatents = S2List(encoded_styles)

    def GenerateImg(self, s_latents):
        with torch.no_grad():
            imgs = self.G.synthesis(
                None, encoded_styles=s_latents, noise_mode="const"
            ).detach().cpu()
            
        return imgs

    def ShowImg(self, num_img=10):
        codes = []
        for i in range(len(self.dlatents)):
            tmp = self.dlatents[i][:num_img, None, :]
            codes.append(tmp)
        out = self.GenerateImg(codes)
        return out

    def SetGParameters(self):
        self.num_layers = self.G.synthesis.num_ws
        self.img_size = self.G.synthesis.img_resolution
        self.s_names = self.GetSName()

        self.img_size = self.G.synthesis.block_resolutions[-1]

        self.mindexs = [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24]

    def MSCode(self, s_latents, boundary):
        manipulated = deepcopy(s_latents)

        for i in range(len(self.s_names)):
            key = self.s_names[i]
            manipulated[key] += self.alpha[0] * torch.from_numpy(boundary[i]).to(manipulated[key])
        
        return manipulated

    def EditOne(self, bname, dlatent_tmp=None):
        if dlatent_tmp == None:
            dlatent_tmp = [
                tmp[self.img_index : (self.img_index + self.num_images)]
                for tmp in self.dlatents
            ]

        boundary_tmp = []
        for i in range(len(self.boundary)):
            tmp = self.boundary[i]
            if len(tmp) <= bname:
                boundary_tmp.append([])
            else:
                boundary_tmp.append(tmp[bname])

        codes = self.MSCode(dlatent_tmp, boundary_tmp)

        out = self.GenerateImg(codes)
        return codes, out

    def EditOneC(self, cindex, dlatent_tmp=None):
        if dlatent_tmp == None:
            dlatent_tmp = [
                tmp[self.img_index : (self.img_index + self.num_images)]
                for tmp in self.dlatents
            ]

        boundary_tmp = [[] for i in range(len(self.dlatents))]

        #'only manipulate 1 layer and one channel'
        assert len(self.manipulate_layers) == 1

        ml = self.manipulate_layers[0]
        tmp = dlatent_tmp[ml].shape[1]  # ada
        tmp1 = np.zeros(tmp)
        tmp1[cindex] = self.code_std[ml][cindex]  # 1
        boundary_tmp[ml] = tmp1

        codes = self.MSCode(dlatent_tmp, boundary_tmp)
        out = self.GenerateImg(codes)
        return codes, out

    def GetFindex(self, lindex, cindex, ignore_RGB=False):

        if ignore_RGB:
            tmp = np.array(self.mindexs) < lindex
            tmp = np.sum(tmp)
        else:
            tmp = lindex
        findex = np.sum(self.fmaps[:tmp]) + cindex
        return findex

    def GetLCIndex(self, findex):
        l_p = []
        cfmaps = np.cumsum(self.fmaps)
        for i in range(len(findex)):
            #    i=-2
            tmp_index = findex[i]
            #    importance_matrix.max(axis=0)
            #    self.attrib_indices2
            tmp = tmp_index - cfmaps
            tmp = tmp[tmp > 0]
            lindex = len(tmp)
            if lindex == 0:
                cindex = tmp_index
            else:
                cindex = tmp[-1]

            if cindex == self.fmaps[lindex]:
                cindex = 0
                lindex += 1
            #        print(completeness.index[i],completeness.iloc[i,:].values,lindex,cindex)
            l_p.append([lindex, cindex])
        l_p = np.array(l_p)
        return l_p

    def GetLCIndex2(self, findex):  # input findex without ToRGB
        fmaps_o = copy.copy(self.fmaps)
        mindexs = [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24]
        self.fmaps = fmaps_o[mindexs]

        l_p = self.GetLCIndex(findex)

        l = l_p[:, 0]
        l2 = np.array(mindexs)[l]
        l_p[:, 0] = l2
        self.fmaps = fmaps_o
        return l_p

    def GetCodeMS(self):
        m = []
        std = []
        for i in range(len(self.dlatents)):
            tmp = self.dlatents[i]
            tmp_mean = tmp.mean(axis=0)
            tmp_std = tmp.std(axis=0)
            m.append(tmp_mean)
            std.append(tmp_std)

        self.code_mean = m
        self.code_std = std
        # return m,std
