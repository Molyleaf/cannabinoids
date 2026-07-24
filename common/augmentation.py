import math
import random
import torch


class SpectrumAugmentation:
    """
    统一的 GC-EI 质谱数据增强流 (全量支持 PyTorch CPU/GPU Tensor 批次处理)
    
    支持三种工作模式：
    - 'pretrain': 预训练全量增强模式（包含强度抖动、m/z 偏移、残基峰移除、瑞利基线与高斯噪声簇）
    - 'finetune_train': 后训练轻量增强模式（轻微强度抖动与基线噪声，防止微调过拟合）
    - 'eval': 评估/推理模式（恒等映射，不进行任何增强变化）
    """
    def __init__(self,
                 mode='pretrain',
                 # 预训练强度抖动与重归一化
                 pretrain_jitter_range=(0.6, 1.5),
                 mz_shift_range=1,
                 minor_peak_threshold_ratio=0.20,
                 minor_peak_removal_prob=0.30,
                 min_peaks_retain=3,
                 rayleigh_scale_tolerance=(0.005, 0.05),
                 n_clusters_range=(1, 5),
                 cluster_width_range=(3, 11),
                 cluster_amp_ratio_range=(0.10, 0.50),
                 # 微调轻量增强参数
                 finetune_jitter_range=(0.85, 1.15),
                 finetune_rayleigh_scale=(0.001, 0.01)
                ):
        self.mode = mode
        self.pretrain_jitter_range = pretrain_jitter_range
        self.mz_shift_range = mz_shift_range
        self.minor_peak_threshold_ratio = minor_peak_threshold_ratio
        self.minor_peak_removal_prob = minor_peak_removal_prob
        self.min_peaks_retain = min_peaks_retain
        self.rayleigh_scale_tolerance = rayleigh_scale_tolerance
        self.n_clusters_range = n_clusters_range
        self.cluster_width_range = cluster_width_range
        self.cluster_amp_ratio_range = cluster_amp_ratio_range
        
        self.finetune_jitter_range = finetune_jitter_range
        self.finetune_rayleigh_scale = finetune_rayleigh_scale

    def __call__(self, spectrum):
        if self.mode == 'eval':
            return spectrum
            
        if spectrum.dim() == 1:
            return self._augment_single(spectrum)
        else:
            return torch.stack([self._augment_single(s) for s in spectrum])

    def _augment_single(self, spec):
        device = spec.device
        
        # 1. 还原到线性强度空间 (平方)
        spec = spec ** 2
        
        if self.mode == 'pretrain':
            # 维度 1: 强度抖动 + 重归一化到 999
            spec = self._intensity_jitter_with_renorm(spec, self.pretrain_jitter_range, device)
            # 维度 2: m/z 整体偏移
            spec = self._mz_shift(spec, device)
            # 维度 3: 低丰度残基峰概率移除
            spec = self._remove_minor_peaks(spec, device)
            # 维度 4: 瑞利 + 高斯噪声注入
            spec = self._inject_noise(spec, device)
        elif self.mode == 'finetune_train':
            # 后训练轻量增强：轻微强度抖动与微弱基线噪声
            spec = self._intensity_jitter(spec, self.finetune_jitter_range, device)
            spec = self._inject_rayleigh_noise(spec, self.finetune_rayleigh_scale, device)
            
        # 重新进行 TIC 归一化 + 平方根缩放
        return self._re_normalize(spec)

    def _intensity_jitter_with_renorm(self, spec, jitter_range, device):
        nonzero_mask = spec > 1e-8
        if nonzero_mask.sum() == 0:
            return spec
            
        jitter_factors = torch.empty(spec.shape[0], device=device).uniform_(
            jitter_range[0], jitter_range[1]
        )
        spec = spec * jitter_factors
        
        spec_max = spec.max()
        if spec_max > 1e-8:
            spec = spec / spec_max * 999.0
        return spec

    def _intensity_jitter(self, spec, jitter_range, device):
        jitter_factors = torch.empty(spec.shape[0], device=device).uniform_(
            jitter_range[0], jitter_range[1]
        )
        return spec * jitter_factors

    def _mz_shift(self, spec, device):
        shift = random.randint(-self.mz_shift_range, self.mz_shift_range)
        if shift != 0:
            spec = torch.roll(spec, shifts=shift, dims=0)
            if shift > 0:
                spec[:shift] = 0.0
            elif shift < 0:
                spec[shift:] = 0.0
        return spec

    def _remove_minor_peaks(self, spec, device):
        base_peak = spec.max()
        if base_peak < 1e-8:
            return spec
            
        dynamic_threshold = base_peak * self.minor_peak_threshold_ratio
        minor_mask = (spec > 1e-8) & (spec < dynamic_threshold)
        
        if minor_mask.any():
            removal_probs = torch.rand(spec.shape[0], device=device)
            removal_mask = minor_mask & (removal_probs < self.minor_peak_removal_prob)
            
            n_total_before = (spec > 1e-8).sum().item()
            n_remove = removal_mask.sum().item()
            
            if n_total_before - n_remove >= self.min_peaks_retain:
                spec[removal_mask] = 0.0
                
        return spec

    def _inject_rayleigh_noise(self, spec, scale_tolerance, device):
        spec_max = spec.max()
        if spec_max < 1e-8:
            return spec
            
        tic = spec.sum()
        rayleigh_scale = random.uniform(
            scale_tolerance[0], scale_tolerance[1]
        ) * tic / spec.shape[0]
        
        rayleigh_noise = torch.empty(spec.shape[0], device=device).exponential_()
        rayleigh_noise = torch.sqrt(rayleigh_noise) * rayleigh_scale
        return torch.clamp(spec + rayleigh_noise, min=0.0)

    def _inject_noise(self, spec, device):
        spec_max = spec.max()
        if spec_max < 1e-8:
            return spec
            
        spec = self._inject_rayleigh_noise(spec, self.rayleigh_scale_tolerance, device)
        
        # 高斯噪声簇
        n_clusters = random.randint(self.n_clusters_range[0], self.n_clusters_range[1])
        for _ in range(n_clusters):
            cluster_center = random.randint(0, spec.shape[0] - 1)
            cluster_width = random.randint(self.cluster_width_range[0], self.cluster_width_range[1])
            
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
            
        return torch.clamp(spec, min=0.0)

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
        return torch.sqrt(spec)
