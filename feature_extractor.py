"""
LogoSP 3D点云特征提取器
使用蒸馏好的预训练模型从点云提取384维特征

使用方法:
    1. 修改下方参数配置
    2. 直接运行此文件
"""

import os
import sys
import torch
import numpy as np
import MinkowskiEngine as ME

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.fpn import Res16FPN18


# ==================== 参数配置 ====================
class Config:
    # 预训练checkpoint路径 (选择对应数据集的蒸馏模型)
    checkpoint_path = '/home/pbw/data1/3D_PointCloud_Segmentation/PLSG_Net/Model_Code/src/LogoSP/LogoSP_ckpt/S3DIS/distill/checkpoint_700.tar'

    # 体素大小 (与训练时保持一致)
    voxel_size = 0.05

    # 输入通道数 (颜色RGB)
    in_channels = 3

    # 输出特征维度
    feat_dim = 384

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # S3DIS数据集配置
    s3dis_root = '/home/pbw/data1/3D_PointCloud_Segmentation/PLSG_Net/dataset/S3DIS/Stanford_Large-Scale_Indoor_Spaces_3D_Dataset/Stanford3dDataset_v1.2_Aligned_Version'
    test_s3dis_scene = 'Area_1/conferenceRoom_1'  # 测试场景 (格式: Area_X/room_name)


# ==================== 特征提取器类 ====================
class LogoSPFeatureExtractor:
    """
    LogoSP 3D点云特征提取器
    
    将点云输入转换为每个点的384维特征向量
    """
    
    def __init__(self, config):
        """
        初始化特征提取器
        
        Args:
            config: 配置对象，包含checkpoint_path, voxel_size等参数
        """
        self.config = config
        self.voxel_size = config.voxel_size
        self.device = config.device
        self.feat_dim = config.feat_dim
        
        # 初始化模型
        self.model = self._build_model()
        self._load_checkpoint()
        self.model.eval()
        
        print(f"[特征提取器] 初始化完成")
        print(f"  - 设备: {self.device}")
        print(f"  - 体素大小: {self.voxel_size}")
        print(f"  - 特征维度: {self.feat_dim}")
    
    def _build_model(self):
        """构建模型"""
        model = Res16FPN18(
            in_channels=self.config.in_channels,
            out_channels=20,  # 这个参数在FPN forward中会被覆盖
            conv1_kernel_size=5,
            config=self.config
        )
        return model.to(self.device)
    
    def _load_checkpoint(self):
        """加载预训练权重"""
        checkpoint_path = self.config.checkpoint_path
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")
        
        print(f"[特征提取器] 加载checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 处理不同格式的checkpoint
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict)
        print(f"[特征提取器] 权重加载成功")
    
    def _preprocess(self, coords, colors=None):
        """
        预处理点云数据
        
        Args:
            coords: (N, 3) numpy array, 点云坐标
            colors: (N, 3) numpy array, RGB颜色 [0-255]，可选
            
        Returns:
            coords: 中心化后的坐标
            feats: 归一化后的颜色特征
        """
        coords = coords.astype(np.float32)
        
        # 中心化坐标
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0  # 保持z轴不变
        coords = coords - coords_center
        
        # 处理颜色特征
        if colors is None:
            # 如果没有颜色，使用零值
            colors = np.zeros_like(coords)
        else:
            colors = colors.astype(np.float32)
            # 归一化到 [-0.5, 0.5]
            colors = colors / 255.0 - 0.5
        
        return coords, colors
    
    def _voxelize(self, coords, feats):
        """
        体素化点云
        
        Args:
            coords: (N, 3) 预处理后的坐标
            feats: (N, 3) 预处理后的特征
            
        Returns:
            voxel_coords: 体素坐标
            voxel_feats: 体素特征
            unique_map: 体素到原始点的映射
            inverse_map: 原始点到体素的映射
        """
        scale = 1.0 / self.voxel_size
        quantized_coords = np.floor(coords * scale)
        
        # 使用MinkowskiEngine进行稀疏量化
        voxel_coords, unique_map, inverse_map = ME.utils.sparse_quantize(
            np.ascontiguousarray(quantized_coords),
            return_index=True,
            return_inverse=True
        )
        
        voxel_feats = feats[unique_map]
        
        return voxel_coords.numpy(), voxel_feats, unique_map, inverse_map
    
    def extract_features(self, coords, colors=None, return_voxel_only=False):
        """
        提取点云特征 (主要接口)
        
        Args:
            coords: (N, 3) numpy array, 点云坐标
            colors: (N, 3) numpy array, RGB颜色 [0-255]，可选
            return_voxel_only: 是否只返回体素级特征 (默认False，返回点级特征)
            
        Returns:
            features: 特征向量
                - return_voxel_only=True: (M, 384) 体素级特征，M为体素数量
                - return_voxel_only=False: (N, 384) 点级特征，N为原始点数量
            info: 包含额外信息的字典
                - voxel_coords: 体素坐标
                - unique_map: 体素到原始点的映射
                - inverse_map: 原始点到体素的映射
        """
        # 预处理
        coords, feats = self._preprocess(coords, colors)
        
        # 体素化
        voxel_coords, voxel_feats, unique_map, inverse_map = self._voxelize(coords, feats)
        
        num_voxels = len(voxel_coords)
        print(f"[特征提取] 原始点数: {len(coords)}, 体素数: {num_voxels}")
        
        # 构建MinkowskiEngine输入
        # 添加batch维度 (batch_id=0)
        batch_coords = np.hstack([
            np.zeros((num_voxels, 1), dtype=np.float32),
            voxel_coords.astype(np.float32)
        ])
        
        coords_tensor = torch.from_numpy(batch_coords).to(self.device)
        feats_tensor = torch.from_numpy(voxel_feats).float().to(self.device)
        
        # 前向传播
        with torch.no_grad():
            in_field = ME.TensorField(feats_tensor, coords_tensor)
            voxel_features = self.model(in_field)
        
        voxel_features = voxel_features.cpu().numpy()
        
        # 构建返回信息
        info = {
            'voxel_coords': voxel_coords,
            'unique_map': unique_map,
            'inverse_map': inverse_map,
            'num_voxels': num_voxels,
            'num_points': len(coords)
        }
        
        if return_voxel_only:
            return voxel_features, info
        else:
            # 将体素特征映射回原始点
            point_features = voxel_features[inverse_map]
            return point_features, info
    
    def extract_features_from_s3dis(self, txt_path, return_voxel_only=False):
        """
        从S3DIS txt文件提取特征
        
        Args:
            txt_path: S3DIS txt文件路径 (格式: x y z r g b 每行)
            return_voxel_only: 是否只返回体素级特征
            
        Returns:
            features: 特征向量
            info: 额外信息字典，包含原始坐标、颜色等
        """
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"S3DIS文件不存在: {txt_path}")
        
        print(f"[特征提取] 读取S3DIS文件: {txt_path}")
        
        # 读取txt文件 (格式: x y z r g b)
        data = np.loadtxt(txt_path)
        
        coords = data[:, :3].astype(np.float32)
        colors = data[:, 3:6].astype(np.uint8)
        
        print(f"[特征提取] 加载点数: {len(coords)}")
        
        # 提取特征
        features, info = self.extract_features(coords, colors, return_voxel_only)
        
        # 添加原始数据到info
        info['coords'] = coords
        info['colors'] = colors
        info['txt_path'] = txt_path
        
        return features, info
    
    def summarize_features(self, features):
        """打印特征统计信息"""
        print(f"[统计] 特征形状: {features.shape}")
        print(f"[统计] 值范围: [{features.min():.4f}, {features.max():.4f}]")
        print(f"[统计] 均值: {features.mean():.4f}, 标准差: {features.std():.4f}")


def run_s3dis_extraction():
    """运行S3DIS特征提取并打印统计信息"""
    config = Config()
    if not config.s3dis_root or not config.test_s3dis_scene:
        raise ValueError("请在Config中设置s3dis_root和test_s3dis_scene")

    scene_name = config.test_s3dis_scene.split('/')[-1]
    txt_path = os.path.join(config.s3dis_root, config.test_s3dis_scene, f"{scene_name}.txt")
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"S3DIS文件不存在: {txt_path}")

    print("=" * 60)
    print("LogoSP 3D点云特征提取器")
    print("=" * 60)
    print(f"[信息] 场景: {config.test_s3dis_scene}")
    print(f"[信息] 点云文件: {txt_path}")

    extractor = LogoSPFeatureExtractor(config)
    features, info = extractor.extract_features_from_s3dis(txt_path)

    print("\n[结果]")
    print(f"  - 输入点数: {info['num_points']:,}")
    print(f"  - 体素数量: {info['num_voxels']:,}")
    extractor.summarize_features(features)

    return features, info


if __name__ == '__main__':
    run_s3dis_extraction()