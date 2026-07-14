import torch
import random
import math

class SpectrumAugmentation:
    """
    GC-EI质谱数据增强（按需求规范设计）
    
    包含四个维度：
    1. 强度抖动 + 重归一化到999
    2. m/z偏移 ±1 Da
    3. 低丰度峰概率性移除
    4. 差异化噪声注入（瑞利基线噪声 + 高斯噪声簇）
    """
    def __init__(self,
                 # 强度抖动
                 intensity_jitter_range=(0.6, 1.5),
                 # m/z偏移
                 mz_shift_range=1,
                 # 低丰度峰移除
                 minor_peak_threshold_ratio=0.20,  # 基峰的20%
                 minor_peak_removal_prob=0.30,     # 移除概率30%
                 min_peaks_retain=3,               # 至少保留3个峰
                 # 瑞利基线噪声
                 rayleigh_scale_tolerance=(0.005, 0.05),  # TIC的0.5%-5%
                 # 高斯噪声簇
                 n_clusters_range=(1, 5),           # 噪声簇数量1-5
                 cluster_width_range=(3, 11),       # 簇宽度3-11 Da
                 cluster_amp_ratio_range=(0.10, 0.50)  # 相邻峰高度的10%-50%
                ):
        self.intensity_jitter_range = intensity_jitter_range
        self.mz_shift_range = mz_shift_range
        self.minor_peak_threshold_ratio = minor_peak_threshold_ratio
        self.minor_peak_removal_prob = minor_peak_removal_prob
        self.min_peaks_retain = min_peaks_retain
        self.rayleigh_scale_tolerance = rayleigh_scale_tolerance
        self.n_clusters_range = n_clusters_range
        self.cluster_width_range = cluster_width_range
        self.cluster_amp_ratio_range = cluster_amp_ratio_range
    
    def __call__(self, spectrum):
        """支持单张和批次处理"""
        if spectrum.dim() == 1:
            return self._augment_single(spectrum)
        else:
            return torch.stack([self._augment_single(s) for s in spectrum])
    
    def _augment_single(self, spec):
        """
        单张谱图增强
        输入: spec (torch.Tensor), 已TIC归一化+sqrt处理过的谱图
        输出: 增强后的谱图
        """
        device = spec.device
        
        # ===== 0. 还原到线性强度空间 (平方) =====
        # 输入的数据是 sqrt 后的，若直接应用强度抖动和噪声注入会产生双重根号效应，动态范围被压缩
        spec = spec ** 2
        
        # ===== Dimension 1: 强度抖动 + 重归一化到999 =====
        spec = self._intensity_jitter_with_renorm(spec, device)
        
        # ===== Dimension 2: m/z 偏移 =====
        spec = self._mz_shift(spec, device)
        
        # ===== Dimension 3: 低丰度峰概率性移除 =====
        spec = self._remove_minor_peaks(spec, device)
        
        # ===== Dimension 4: 差异化噪声注入 =====
        spec = self._inject_noise(spec, device)
        
        # ===== 后处理：重新进行TIC归一化 + sqrt =====
        spec = self._re_normalize(spec)
        
        return spec
    
    def _intensity_jitter_with_renorm(self, spec, device):
        """强度抖动 + 重归一化到999"""
        nonzero_mask = spec > 1e-8
        
        if nonzero_mask.sum() == 0:
            return spec
        
        # 为每个非零点生成随机波动因子 [0.6, 1.5]
        jitter_factors = torch.empty(spec.shape[0], device=device).uniform_(
            self.intensity_jitter_range[0],
            self.intensity_jitter_range[1]
        )
        
        # 应用波动
        spec = spec * jitter_factors
        
        # 找到最大峰，重归一化到999
        spec_max = spec.max()
        if spec_max > 1e-8:
            spec = spec / spec_max * 999.0
        
        return spec
    
    def _mz_shift(self, spec, device):
        """m/z整体偏移 ±1 Da"""
        shift = random.randint(-self.mz_shift_range, self.mz_shift_range)
        if shift != 0:
            spec = torch.roll(spec, shifts=shift, dims=0)
            if shift > 0:
                spec[:shift] = 0.0
            elif shift < 0:
                spec[shift:] = 0.0
        return spec
    
    def _remove_minor_peaks(self, spec, device):
        """
        低丰度峰概率性移除
        """
        base_peak = spec.max()
        if base_peak < 1e-8:
            return spec
        
        dynamic_threshold = base_peak * self.minor_peak_threshold_ratio
        
        # 找到低于阈值的非零点
        minor_mask = (spec > 1e-8) & (spec < dynamic_threshold)
        
        if minor_mask.any():
            # 对每个低丰度峰，以30%概率移除
            removal_probs = torch.rand(spec.shape[0], device=device)
            removal_mask = minor_mask & (removal_probs < self.minor_peak_removal_prob)
            
            # 检查移除后总峰数是否≥3
            n_total_before = (spec > 1e-8).sum().item()
            n_remove = removal_mask.sum().item()
            
            if n_total_before - n_remove >= self.min_peaks_retain:
                spec[removal_mask] = 0.0
        
        return spec
    
    def _inject_noise(self, spec, device):
        """
        差异化噪声注入：
        1. 瑞利基线噪声（整个谱图）
        2. 高斯噪声簇（局部m/z区间）
        """
        spec_max = spec.max()
        if spec_max < 1e-8:
            return spec
        
        tic = spec.sum()
        
        # ----- 瑞利基线噪声 -----
        rayleigh_scale = random.uniform(
            self.rayleigh_scale_tolerance[0],
            self.rayleigh_scale_tolerance[1]
        ) * tic / spec.shape[0]
        
        rayleigh_noise = torch.empty(spec.shape[0], device=device).exponential_()
        rayleigh_noise = torch.sqrt(rayleigh_noise) * rayleigh_scale
        
        spec = spec + rayleigh_noise
        
        # ----- 高斯噪声簇 -----
        n_clusters = random.randint(
            self.n_clusters_range[0],
            self.n_clusters_range[1]
        )
        
        for _ in range(n_clusters):
            cluster_center = random.randint(0, spec.shape[0] - 1)
            cluster_width = random.randint(
                self.cluster_width_range[0],
                self.cluster_width_range[1]
            )
            
            start = max(0, cluster_center - cluster_width // 2)
            end = min(spec.shape[0], cluster_center + cluster_width // 2 + 1)
            
            local_peaks = spec[start:end]
            local_max = local_peaks.max()
            
            amp_ratio = self._sample_lognormal(
                self.cluster_amp_ratio_range[0],
                self.cluster_amp_ratio_range[1],
                device
            )
            cluster_amplitude = local_max * amp_ratio
            
            positions = torch.arange(start, end, device=device, dtype=torch.float32)
            center_pos = cluster_center
            sigma = cluster_width / 6.0
            
            gaussian_peak = cluster_amplitude * torch.exp(
                -0.5 * ((positions - center_pos) / sigma) ** 2
            )
            
            spec[start:end] = spec[start:end] + gaussian_peak
        
        spec = torch.clamp(spec, min=0.0)
        return spec
    
    def _sample_lognormal(self, low, high, device):
        mean = (low + high) / 2
        sigma = (high - low) / 4
        log_mean = math.log(mean)
        log_sigma = sigma / mean
        
        sample = torch.empty(1, device=device).log_normal_(
            mean=log_mean, std=log_sigma
        )
        return torch.clamp(sample, low, high).item()
    
    def _re_normalize(self, spec):
        tic = spec.sum()
        if tic > 1e-8:
            spec = spec / tic
        # 直接使用 torch.sqrt 避免抬高原本为 0 的背景通道，保留高稀疏性质
        return torch.sqrt(spec)
