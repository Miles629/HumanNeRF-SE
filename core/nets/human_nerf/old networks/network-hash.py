import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils.network_util import MotionBasisComputer

from core.nets.human_nerf.embedders.hash_encoding import get_hashencoding,get_shencoding
import core.nets.human_nerf.embedders.encoding as encoding

# from core.nets.human_nerf.canonical_mlps.NerfSmall import NeRFSmall
from core.nets.human_nerf.canonical_mlps.smallMLP import CanonicalMLP

from configs import cfg
import spconv.pytorch as spconv
# from core.nets.human_nerf.embedders import embedder
# from core.nets.human_nerf.sparseconv import SparseConvNet
from third_parties.smpl.smpl_numpy import SMPL
from core.data.dataset_args import DatasetArgs

def load_canonical_joints(dataset_path):
        cl_joint_path = os.path.join(dataset_path, 'canonical_joints.pkl')
        with open(cl_joint_path, 'rb') as f:
            cl_joint_data = pickle.load(f)
        canonical_joints = cl_joint_data['joints'].astype('float32')
        canonical_bbox = skeleton_to_bbox(canonical_joints)
        return canonical_joints, canonical_bbox
def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + cfg.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # self.c = nn.Embedding(6890, 16)
        self.xyzc_net = SparseConvNet()
        self.code = self.get_code().to(cfg.primary_gpus[0])
        self.code.requires_grad = False

        # get hash encoding fuction
        args = DatasetArgs.get(cfg.train.dataset)
        canonical_joints, canonical_bbox = load_canonical_joints(args['dataset_path'])
        self.canonical_bbox=torch.Tensor([canonical_bbox['min_xyz'],canonical_bbox['max_xyz']]).to(cfg.primary_gpus[0])
        # self.hash_fn, hash_ch = get_hashencoding(canonical_bbox,log2_hashmap_size=19,finest_res=1024,n_levels=16)
        # self.hash_fn.to(cfg.primary_gpus[0])
        self.hash_fn = encoding.MultiResHashGrid(3)
        # small nerf
        # self.smallNerf = NeRFSmall(num_layers=3,
        #                 hidden_dim=64,
        #                 geo_feat_dim=3,
        #                 num_layers_color=3,
        #                 hidden_dim_color=64,
        #                 input_ch=32).to(cfg.primary_gpus[0])
        self.smallNerf = CanonicalMLP(mlp_depth=4, mlp_width=128, 
                 input_ch=32, skips=None,)
        






        # self.latent = nn.Embedding(cfg.max_frame, 128)
        
        # motion basis computer
        self.motion_basis_computer = MotionBasisComputer(
                                        total_bones=cfg.total_bones)

        # motion weight volume
        # self.mweight_vol_decoder = load_mweight_vol_decoder(cfg.mweight_volume.module)(
        #     embedding_size=cfg.mweight_volume.embedding_size,
        #     volume_size=cfg.mweight_volume.volume_size,
        #     total_bones=cfg.total_bones
        # )

        # # non-rigid motion st positional encoding
        # self.get_non_rigid_embedder = \
        #     load_positional_embedder(cfg.non_rigid_embedder.module)

        # # non-rigid motion MLP
        # _, non_rigid_pos_embed_size = \
        #     self.get_non_rigid_embedder(cfg.non_rigid_motion_mlp.multires, 
        #                                 cfg.non_rigid_motion_mlp.i_embed)
        # self.non_rigid_mlp = \
        #     load_non_rigid_motion_mlp(cfg.non_rigid_motion_mlp.module)(
        #         pos_embed_size=non_rigid_pos_embed_size,
        #         condition_code_size=cfg.non_rigid_motion_mlp.condition_code_size,
        #         mlp_width=cfg.non_rigid_motion_mlp.mlp_width,
        #         mlp_depth=cfg.non_rigid_motion_mlp.mlp_depth,
        #         skips=cfg.non_rigid_motion_mlp.skips)
        # self.non_rigid_mlp = \
        #     nn.DataParallel(
        #         self.non_rigid_mlp,
        #         device_ids=cfg.secondary_gpus,
        #         output_device=cfg.secondary_gpus[0])

        # canonical positional encoding
        # get_embedder = load_positional_embedder(cfg.embedder.module)

        # cnl_pos_embed_fn, cnl_pos_embed_size = \
        #     get_embedder(cfg.canonical_mlp.multires, 
        #                  cfg.canonical_mlp.i_embed)
        # self.pos_embed_fn = cnl_pos_embed_fn

        # # pose positional encoding
        # get_embedder = load_positional_embedder(cfg.embedder.module)
        


        # # canonical mlp 
        # skips = [4]
        # self.cnl_mlp = \
        #     load_canonical_mlp(cfg.canonical_mlp.module)(
        #         input_ch=cnl_pos_embed_size, 
        #         mlp_depth=cfg.canonical_mlp.mlp_depth, 
        #         mlp_width=cfg.canonical_mlp.mlp_width,
        #         skips=skips)
        # self.cnl_mlp = \
        #     nn.DataParallel(
        #         self.cnl_mlp,
        #         device_ids=cfg.secondary_gpus,
        #         output_device=cfg.primary_gpus[0])

        # # pose decoder MLP
        # self.pose_decoder = \
        #     load_pose_decoder(cfg.pose_decoder.module)(
        #         embedding_size=cfg.pose_decoder.embedding_size,
        #         mlp_width=cfg.pose_decoder.mlp_width,
        #         mlp_depth=cfg.pose_decoder.mlp_depth)
    def get_code(self):
            modeldir = './third_parties/smpl/models'
            smpl_model = SMPL(sex='neutral', model_dir=modeldir)
            weight = torch.tensor(smpl_model.get_weight(),dtype=torch.float32)
            code = torch.ones(6890,1)
            # print(weight,code)
            # exit()
            output = torch.cat([code,weight],dim=1)
            return output
    def verticesfiltertorch(self, pos_flat, **kwargs):
        out_sh = kwargs['out_sh']
        code = torch.ones(6890).to(pos_flat.device)
        vertices = kwargs['vertices']
        pts_voxel = (pos_flat- kwargs['dst_bbox'][0])/ torch.tensor(cfg.voxelsize).to(pos_flat.device)
      
        pts_voxel = pts_voxel / (out_sh-1) * 2 - 1
        pts_voxel = pts_voxel[...,[2,1,0]].float()

# #      debug
#         pts_voxel = vertices[...,[2,1,0]]
#         # pts_voxel[0]= pts_voxel[0]-1
#         pts_voxel = pts_voxel/(out_sh-1) * 2 - 1
#         pts_voxel = pts_voxel.float()

        pts_voxel = pts_voxel[None,None,None,...]
        # print("pts_voxel",pts_voxel)

        out_sh = out_sh.tolist()
        # print("vertices",vertices,vertices.shape)
        volume = torch.sparse_coo_tensor(vertices.t(),code,out_sh)
        # print("volume",volume,volume.shape)
        volume = volume.to_dense()
        # print(torch.where(volume==1)[2].shape,torch.where(volume!=0)[1].shape)

        # 
        volume = volume[None,None,...]
        # pts_voxel = pts_voxel.transpose(1,0)
        # pts_voxel = pts_voxel[None,...]

        # voxelvalueinterplote = F.interpolate(volume,tuple(pts_voxel.tolist()))
        # print("voxelvalueinterplote",voxelvalueinterplote)
        # exit()

        voxelvalue = F.grid_sample(volume,
                                    pts_voxel,
                                    mode='bilinear',
                                    padding_mode='zeros',
                                    align_corners=True)

        # print("voxelvalue",voxelvalue.shape)
        voxelvalue = voxelvalue.view(voxelvalue.shape[-1])
        # print("voxelvalue",voxelvalue,voxelvalue.shape)
        # print(torch.where(voxelvalue!=0)[0].shape)
        nonzero_indices = torch.nonzero(voxelvalue).view(-1)
        # print("pts_voxel[nonzero_indices]",pts_voxel0.reshape(pts_voxel0.shape[3],3)[nonzero_indices])
        # print("nonzero_indices",nonzero_indices.size())
        # print(voxelvalue[nonzero_indices])
        # exit()
        return nonzero_indices


    def verticesfilter(self, pos_flat, **kwargs):
        # pos_flat = torch.reshape(pts, [-1, pts.shape[-1]])
        chunk = cfg.netchunk_per_gpu*len(cfg.secondary_gpus)
        # print("pos_flat",pos_flat.shape)
        vertices = kwargs['vertices']
        out_sh = kwargs['out_sh']
        # code = (torch.arange(1, 6891)/6891).to(pos_flat.device)
        # code = code.view(6890,1)

        # print(pts.shape)
        # out_sh = torch.tensor(out_sh).to(pos_flat.device)
        # xyz = pos_flat[start:end]
        pts_voxel = pos_flat.reshape(-1,3)
        # print("kwargs['dst_bbox']",kwargs['dst_bbox'][0],kwargs['dst_bbox'][1])
        pts_voxel = (pts_voxel- kwargs['dst_bbox'][0])/ torch.tensor(cfg.voxelsize).to(pos_flat.device)
        # out_sh = torch.tensor(out_sh).to(pts_voxel)
        # print("pts_voxel",pts_voxel0,pts_voxel0.shape)
        # exit()
        pts_voxel = pts_voxel / (out_sh-1) * 2 - 1
        pts_voxel = pts_voxel[...,[2,1,0]].float()
        pts_voxel = pts_voxel[None,None,None,...]

      #debug
        # pts_voxel = vertices / (out_sh-1) * 2 - 1
        # pts_voxel = pts_voxel[...,[2,1,0]].float()
        # pts_voxel = pts_voxel[None,None,None,...]


        code = self.code
        # print(vertices.dtype)
        idx = torch.full([vertices.shape[0]], 0).to(vertices)
        vertices = torch.cat([idx[:, None], vertices], dim=1)
        # print(torch.max(vertices,dim=0))
        # out_sh, _ = torch.max(out_sh, dim=0)
        out_sh = out_sh.tolist()
        # print("out_sh:",out_sh)
        # print('spconv shape:',code.shape,vertices.shape,vertices.dtype)
        xyz_volume=[]
        for i in range(25):
            sp_tensor = spconv.SparseConvTensor(code[...,i].view(6890,1), vertices, out_sh, 1)
            xyz_volume.append(sp_tensor)
        # print("xyzc",xyzc.dense().shape)
        feature_volume = self.xyzc_net(xyz_volume)
        # print(torch.where(xyzc.dense()!=0)[3].shape)
        # print(torch.where(xyzc.dense()==1)[3].shape)

        xyzc_features = self.interpolate_features(pts_voxel,feature_volume)
        # print("xyzc_features",xyzc_features,xyzc_features.shape)([1, 25, 307200])
        xyzc_features = xyzc_features[0,...].transpose(1,0)
        # print("xyzc_features",xyzc_features,xyzc_features.shape)

        nonzero_indices = torch.nonzero(xyzc_features[...,0]).view(-1)
        # print("pts_voxel[nonzero_indices]",pts_voxel0.reshape(pts_voxel0.shape[3],3)[nonzero_indices])
        # print("nonzero_indices",nonzero_indices.size())
        # print(xyzc_features[nonzero_indices])
        # exit()


#       debug
        # feature_volume = [xyzc.dense()]
        # xyzc_features = self.interpolate_features(pts_voxel,feature_volume)
        # xyzc_features = xyzc_features.view(xyzc_features.shape[2])
        # print("xyzc_features",xyzc_features,xyzc_features.shape)
        # print(torch.where(xyzc_features!=0)[0].shape)
        # nonzero_indices = torch.nonzero(xyzc_features).view(-1)
        # # print("pts_voxel[nonzero_indices]",pts_voxel0.reshape(pts_voxel0.shape[3],3)[nonzero_indices])
        # print("nonzero_indices",nonzero_indices.size())
        # print(xyzc_features[nonzero_indices])

        # debug
        # id = kwargs['idx'].item()
        # if id == 135:
        #     pts_voxel = pts_voxel[0][0][0]
        #     print("135 out_sh:",out_sh)
        #     print("vertices max min:",vertices[...,1:].max(dim=0)[0],vertices[...,1:].min(dim=0)[0])
        #     print("pts_voxel max min:",pts_voxel[...,[2,1,0]].max(dim=0)[0],pts_voxel[...,[2,1,0]].min(dim=0)[0])
        #     pts_voxel = (pts_voxel[...,[2,1,0]]+1)/2*(torch.tensor(out_sh).to(pts_voxel.device)-1)
        #     print("pts_voxel max min:",pts_voxel.max(dim=0)[0],pts_voxel.min(dim=0)[0])
        #     if nonzero_indices.shape[0]==0:
        #         print("135 points not found in vertices space")
        # if pos_flat.shape[0]>270000 and nonzero_indices.shape[0]==0:
        #     pts_voxel = pts_voxel[0][0][0]
        #     print("out_sh:",out_sh)
        #     print("vertices max min:",vertices[...,1:].max(dim=0)[0],vertices[...,1:].min(dim=0)[0])
        #     print("pts_voxel max min:",pts_voxel[...,[2,1,0]].max(dim=0)[0],pts_voxel[...,[2,1,0]].min(dim=0)[0])
        #     pts_voxel = (pts_voxel[...,[2,1,0]]+1)/2*(torch.tensor(out_sh).to(pts_voxel.device)-1)
        #     print("pts_voxel max min:",pts_voxel.max(dim=0)[0],pts_voxel.min(dim=0)[0])


        # exit()
        
        # print("pos_flat[nonzero_indices]shape",pos_flat[nonzero_indices].shape)
        return nonzero_indices,xyzc_features



    def interpolate_features(self, grid_coords, feature_volume):
        features = []
        for volume in feature_volume:
            # print("for volume in feature_volume")
            feature = F.grid_sample(volume,
                                    grid_coords,
                                    mode='nearest',
                                    padding_mode='zeros',
                                    align_corners=False)
            features.append(feature)
        features = torch.cat(features, dim=1)
        features = features.view(features.size(0), -1, features.size(4))
        return features
    
    # def deploy_mlps_to_secondary_gpus(self):
    #     self.cnl_mlp = self.cnl_mlp.to(cfg.secondary_gpus[0])
    #     # self.c = self.c.to(cfg.secondary_gpus[0])
    #     # self.xyzc_net = self.xyzc_net.to(cfg.secondary_gpus[0])
    #     if self.non_rigid_mlp:
    #         self.non_rigid_mlp = self.non_rigid_mlp.to(cfg.secondary_gpus[0])

    #     return self


    def _query_mlp(
            self,
            pos_xyz,
            # non_rigid_pos_embed_fn,
            non_rigid_mlp_input,
            ob_feature=None):

        # (N_rays, N_samples, 3) --> (N_rays x N_samples, 3)
        # pos_flat = torch.reshape(pos_xyz, [-1, pos_xyz.shape[-1]])
        pos_flat = pos_xyz
        chunk = cfg.netchunk_per_gpu*len(cfg.secondary_gpus)

        result = self._apply_mlp_kernals(
                        pos_flat=pos_flat,
                        chunk=chunk)

        output = {}

        raws_flat = result['raws']
        output['raws'] = torch.reshape(
                            raws_flat, 
                            list(pos_xyz.shape[:-1]) + [raws_flat.shape[-1]])

        return output


    @staticmethod
    def _expand_input(input_data, total_elem):
        assert input_data.shape[0] == 1
        input_size = input_data.shape[1]
        return input_data.expand((total_elem, input_size))


    def _apply_mlp_kernals(
            self, 
            pos_flat,
            chunk,
            ob_feature=None):
        raws = []

        # iterate ray samples by trunks
        for i in range(0, pos_flat.shape[0], chunk):
            start = i
            end = i + chunk
            if end > pos_flat.shape[0]:
                end = pos_flat.shape[0]
            total_elem = end - start

            xyz = pos_flat[start:end]
           

            # if not cfg.ignore_non_rigid_motions:
            #     non_rigid_embed_xyz = non_rigid_pos_embed_fn(xyz)
            #     result = self.non_rigid_mlp(
            #         pos_embed=non_rigid_embed_xyz,
            #         pos_xyz=xyz,
            #         condition_code=condition_code
            #     )
            #     xyz = result['xyz']

            xyzinput = (xyz-self.canonical_bbox[0])/(self.canonical_bbox[1]-self.canonical_bbox[0])
            xyz_embedded = self.hash_fn(xyzinput)
            # xyz_embedded , keep_mask= self.hash_fn(xyz)
            
                
            cnlout = self.smallNerf(pos_embed=xyz_embedded)
            # cnlout[~keep_mask, -1] = 0 # set sigma to 0 for invalid points
            raws += [cnlout]

            # xyz_embedded = pos_embed_fn(xyz)
            # raws += [self.cnl_mlp(xyz_embedded)]
        raws = torch.cat(raws, dim=0)
        output = {}
        output['raws'] = raws

        return output


    def _batchify_rays(self, rays_flat, **kwargs):
        all_ret = {}
        for i in range(0, rays_flat.shape[0], cfg.chunk):
            ret = self._render_rays(rays_flat[i:i+cfg.chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret





    @staticmethod
    def _sample_motion_fields( 
            pts,
            motion_scale_Rs,
            motion_Ts, 
            motion_weights_vol,
            cnl_bbox_min_xyz, cnl_bbox_scale_xyz,
            output_list):
        # orig_shape = list(pts.shape)
        pts = pts.reshape(-1, 3) # [N_rays x N_samples, 3]

        # remove BG channel
        motion_weights = motion_weights_vol[:-1] 
        # print('motion_weights:',motion_weights,motion_weights.shape)torch.Size([24, 32, 32, 32])
        # print('motion_scale_Rs',motion_scale_Rs,motion_scale_Rs.shape)torch.Size([24, 3, 3])

        weights_list = []
        for i in range(motion_weights.size(0)):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            # print("pos骨骼变形",pos.shape) torch.Size([23678, 3])
            pos = (pos - cnl_bbox_min_xyz[None, :]) \
                            * cnl_bbox_scale_xyz[None, :] - 1.0 
            weights = F.grid_sample(input=motion_weights[None, i:i+1, :, :, :], 
                                    grid=pos[None, None, None, :, :],           
                                    padding_mode='zeros', align_corners=True)
            weights = weights[0, 0, 0, 0, :, None] 
            # print("weight",weights.shape)torch.Size([32361, 1])
            weights_list.append(weights) 
        backwarp_motion_weights = torch.cat(weights_list, dim=-1)
        # print("backwarp_motion_weights",backwarp_motion_weights.shape)([23678, 24])
        total_bases = backwarp_motion_weights.shape[-1]
        # print("total_bases",total_bases)24

        backwarp_motion_weights_sum = torch.sum(backwarp_motion_weights, 
                                                dim=-1, keepdim=True)
        # print("backwarp_motion_weights_sum",backwarp_motion_weights_sum.shape)([23678, 1])
        weighted_motion_fields = []
        for i in range(total_bases):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            weighted_pos = backwarp_motion_weights[:, i:i+1] * pos
            # print("weighted_pos",weighted_pos.shape)torch.Size([32361, 3])
            weighted_motion_fields.append(weighted_pos)
        x_skel = torch.sum(
                        torch.stack(weighted_motion_fields, dim=0), dim=0
                        ) / backwarp_motion_weights_sum.clamp(min=0.0001)
        # print("x_skel",x_skel.shape)
        
        fg_likelihood_mask = backwarp_motion_weights_sum

        # x_skel = x_skel.reshape(orig_shape[:2]+[3])
        # # backwarp_motion_weights = \
        # #     backwarp_motion_weights.reshape(orig_shape[:2]+[total_bases])
        # fg_likelihood_mask = fg_likelihood_mask.reshape(orig_shape[:2]+[1])

        results = {}
        
        if 'x_skel' in output_list: # [N_rays x N_samples, 3]
            results['x_skel'] = x_skel
        if 'fg_likelihood_mask' in output_list: # [N_rays x N_samples, 1]
            results['fg_likelihood_mask'] = fg_likelihood_mask
        return results


    @staticmethod
    def _unpack_ray_batch(ray_batch):
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] 
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2]) 
        near, far = bounds[...,0], bounds[...,1] 
        return rays_o, rays_d, near, far


    @staticmethod
    def _get_samples_along_ray(N_rays, near, far):
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        # print(z_vals)
        return z_vals.expand([N_rays, cfg.N_samples]) 


    @staticmethod
    def _stratified_sampling(z_vals):#分层采样
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        
        t_rand = torch.rand(z_vals.shape).to(z_vals)
        z_vals = lower + (upper - lower) * t_rand

        return z_vals


            
    def _render_rays(
            self, 
            ray_batch, 
            motion_scale_Rs,
            motion_Ts,
            # non_rigid_pos_embed_fn,
            non_rigid_mlp_input=None,
            bgcolor=None,
            **kwargs):
        with torch.no_grad():
            N_rays = ray_batch.shape[0]
            rays_o, rays_d, near, far = self._unpack_ray_batch(ray_batch)

            z_vals = self._get_samples_along_ray(N_rays, near, far)
            if cfg.perturb > 0.:
                z_vals = self._stratified_sampling(z_vals)

            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]

            
            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
            
            # ob_raw=self.observation_nerf(pts,rays_d,**kwargs)
            nray,nsam,dims = pts.shape
            pos_flat = torch.reshape(pts, [-1, pts.shape[-1]])
            # nonzero_indices = self.verticesfiltertorch(pos_flat,**kwargs)
            nonzero_indices,weight = self.verticesfilter(pos_flat,**kwargs)
            # print("id",kwargs['idx'],"nonzero_indices",nonzero_indices.shape,"pos_flat",pos_flat.shape)

            if nonzero_indices.shape[0]==0:
                print("id",kwargs['idx'],"nonzero_indices",nonzero_indices.shape,"pos_flat",pos_flat.shape)
                    
                allraw = torch.zeros(pos_flat.shape[0],4).to(pos_flat.device)
                allraw = allraw.view(nray,nsam,4)
                rgb_map, acc_map, _, depth_map = self._raw2outputs_norawmask(allraw, z_vals, rays_d, bgcolor)
                return {'rgb' : rgb_map,  
                        'alpha' : acc_map, 
                        'depth': depth_map}


            ptsfiltered = pos_flat[nonzero_indices]
            weight = weight[nonzero_indices]
            ptsweight = weight[...,1:]/weight[...,0,None]
            # print(torch.where(ptsweight[...,0]!=0)[0].shape,torch.where(ptsweight[...,0]==1.)[0].shape)

            pos_trans =[]
            ptsweight_sum = torch.sum(ptsweight, dim=-1, keepdim=True)
        # print("ptsweight_sum,",ptsweight_sum)
        for i in range(24):
            pos = torch.matmul(motion_scale_Rs[0,i, :, :], ptsfiltered.T).T + motion_Ts[0,i, :]
            weighted_pos = ptsweight[:, i:i+1] * pos
            # print("weighted_pos",weighted_pos.shape)torch.Size([32361, 3])
            pos_trans.append(weighted_pos)
        cnl_pts = torch.sum(
                        torch.stack(pos_trans, dim=0), dim=0
                        ) / ptsweight_sum.clamp(min=0.0001)
        # exit()

        
        # mv_output = self._sample_motion_fields(
        #                     pts=ptsfiltered,
        #                     motion_scale_Rs=motion_scale_Rs[0], 
        #                     motion_Ts=motion_Ts[0], 
        #                     motion_weights_vol=motion_weights_vol,
        #                     cnl_bbox_min_xyz=cnl_bbox_min_xyz, 
        #                     cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
        #                     output_list=['x_skel', 'fg_likelihood_mask'])
        # pts_mask = mv_output['fg_likelihood_mask']
        # cnl_pts = mv_output['x_skel']
        # print("cnl_pts",cnl_pts.shape)
        # print("pts_mask",pts_mask.shape)
     


        query_result = self._query_mlp(
                                pos_xyz=cnl_pts,
                                non_rigid_mlp_input=non_rigid_mlp_input,
                                # non_rigid_pos_embed_fn=non_rigid_pos_embed_fn
                                )
        raw = query_result['raws']
        # print("raw",raw.shape)
        allraw = torch.zeros(pos_flat.shape[0],4).to(pos_flat.device)
        # print("allraw",allraw.shape)
        allraw[nonzero_indices] = raw
        # print("nray",nray,"nsam",nsam)
        allraw = allraw.view(nray,nsam,4)
        # print("allraw",allraw.shape)


        
        rgb_map, acc_map, _, depth_map = self._raw2outputs_norawmask(allraw, z_vals, rays_d, bgcolor)
        # rgb_map, acc_map, _, depth_map = self._raw2outputs(allraw, pts_mask, z_vals, rays_d, bgcolor)
    
        return {'rgb' : rgb_map,  
                'alpha' : acc_map, 
                'depth': depth_map}
    @staticmethod
    def _raw2outputs_norawmask(cnl_raw, z_vals, rays_d, bgcolor=None):
        def _raw2alpha(raw, dists, act_fn=F.relu):
            return 1.0 - torch.exp(-act_fn(raw)*dists)

        dists = z_vals[...,1:] - z_vals[...,:-1]

        infinity_dists = torch.Tensor([1e10])
        infinity_dists = infinity_dists.expand(dists[...,:1].shape).to(dists)
        dists = torch.cat([dists, infinity_dists], dim=-1) 
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        rgb = torch.sigmoid(cnl_raw[...,:3])  # [N_rays, N_samples, 3]
        alpha = _raw2alpha(cnl_raw[...,3], dists)  # [N_rays, N_samples]

        # alpha = alpha * raw_mask[:, :, 0]
        alpha = alpha 
        

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha), 
                       1.-alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        acc_map = torch.sum(weights, -1)
        rgb_map = rgb_map + (1.-acc_map[...,None]) * bgcolor[None, :]/255.
        return rgb_map, acc_map, weights, depth_map
    
    @staticmethod
    def _raw2outputs(cnl_raw, raw_mask, z_vals, rays_d, bgcolor=None):
        def _raw2alpha(raw, dists, act_fn=F.relu):
            return 1.0 - torch.exp(-act_fn(raw)*dists)

        dists = z_vals[...,1:] - z_vals[...,:-1]

        infinity_dists = torch.Tensor([1e10])
        infinity_dists = infinity_dists.expand(dists[...,:1].shape).to(dists)
        dists = torch.cat([dists, infinity_dists], dim=-1) 
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        rgb = torch.sigmoid(cnl_raw[...,:3])  # [N_rays, N_samples, 3]
        alpha = _raw2alpha(cnl_raw[...,3], dists)  # [N_rays, N_samples]

        # alpha = alpha * raw_mask[:, :, 0]
        alpha = alpha 
        

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha), 
                       1.-alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        acc_map = torch.sum(weights, -1)
        rgb_map = rgb_map + (1.-acc_map[...,None]) * bgcolor[None, :]/255.
        return rgb_map, acc_map, weights, depth_map

    def _get_motion_base(self, dst_Rs, dst_Ts, cnl_gtfms):
        motion_scale_Rs, motion_Ts = self.motion_basis_computer(
                                        dst_Rs, dst_Ts, cnl_gtfms)

        return motion_scale_Rs, motion_Ts


    @staticmethod
    def _multiply_corrected_Rs(Rs, correct_Rs):
        total_bones = cfg.total_bones - 1
        return torch.matmul(Rs.reshape(-1, 3, 3),
                            correct_Rs.reshape(-1, 3, 3)).reshape(-1, total_bones, 3, 3)

    
    def forward(self,
                rays, 
                dst_Rs, dst_Ts, cnl_gtfms,
                motion_weights_priors,
                dst_posevec=None,
                near=None, far=None,
                iter_val=1e7,
                **kwargs):

        dst_Rs=dst_Rs[None, ...]
        dst_Ts=dst_Ts[None, ...]
        dst_posevec=dst_posevec[None, ...]
        cnl_gtfms=cnl_gtfms[None, ...]
        motion_weights_priors=motion_weights_priors[None, ...]

        
        # exit()

        # correct body pose
        # if iter_val >= cfg.pose_decoder.get('kick_in_iter', 0):
        #     pose_out = self.pose_decoder(dst_posevec)
        #     refined_Rs = pose_out['Rs']
        #     refined_vec = pose_out['rvec']
        #     # smpl_model(refined_vec,)
        #     refined_Ts = pose_out.get('Ts', None)
            
        #     dst_Rs_no_root = dst_Rs[:, 1:, ...]
        #     dst_Rs_no_root = self._multiply_corrected_Rs(
        #                                 dst_Rs_no_root, 
        #                                 refined_Rs)
        #     dst_Rs = torch.cat(
        #         [dst_Rs[:, 0:1, ...], dst_Rs_no_root], dim=1)

        #     if refined_Ts is not None:
        #         dst_Ts = dst_Ts + refined_Ts

        # non_rigid_pos_embed_fn, _ = \
        #     self.get_non_rigid_embedder(
        #         multires=cfg.non_rigid_motion_mlp.multires,                         
        #         is_identity=cfg.non_rigid_motion_mlp.i_embed,
        #         iter_val=iter_val,)

        if iter_val < cfg.non_rigid_motion_mlp.kick_in_iter:
            # mask-out non_rigid_mlp_input 
            non_rigid_mlp_input = torch.zeros_like(dst_posevec) * dst_posevec
        else:
            non_rigid_mlp_input = dst_posevec

        kwargs.update({
            # "non_rigid_pos_embed_fn": non_rigid_pos_embed_fn,
            "non_rigid_mlp_input": non_rigid_mlp_input
        })

        motion_scale_Rs, motion_Ts = self._get_motion_base(
                                            dst_Rs=dst_Rs, 
                                            dst_Ts=dst_Ts, 
                                            cnl_gtfms=cnl_gtfms)
        # print('motion_weights_vol_origin:',motion_weights_vol,motion_weights_vol.shape)
        # motion_weights_vol = self.mweight_vol_decoder(motion_weights_priors=motion_weights_priors)
        # motion_weights_vol=motion_weights_vol[0] # remove batch dimension
        # print('motion_weights_vol_netoutput:',motion_weights_vol,motion_weights_vol.shape)

        kwargs.update({
            'motion_scale_Rs': motion_scale_Rs,
            'motion_Ts': motion_Ts,
        })

        rays_o, rays_d = rays
        rays_shape = rays_d.shape 

        rays_o = torch.reshape(rays_o, [-1,3]).float()
        rays_d = torch.reshape(rays_d, [-1,3]).float()
        packed_ray_infos = torch.cat([rays_o, rays_d, near, far], -1)

        all_ret = self._batchify_rays(packed_ray_infos, **kwargs)
        for k in all_ret:
            k_shape = list(rays_shape[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_shape)

        return all_ret



class SparseConvNet(nn.Module):
    def __init__(self):
        super(SparseConvNet, self).__init__()
        k=5
        s=1
        p=2

        self.one_conv = spconv.SparseConv3d(1,
                          1,
                          kernel_size=k,
                          stride=s,
                          padding=p,
                        #   groups=25,
                          bias=False,
                          indice_key='cp0')
        self.one_conv.weight.requires_grad = False
        self.one_conv.weight.data = torch.ones_like(self.one_conv.weight.data)
        

    def forward(self, x):
        netout = []
        for i in x:
            net = self.one_conv(i)
            netout.append(net.dense())
        return netout

