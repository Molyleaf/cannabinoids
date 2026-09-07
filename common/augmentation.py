import math
import random
import torch


class SpectrumAugmentation:
    """
    Unified GC-EI mass spectrum data augmentation pipeline (fully GPU Tensor 2D Batch matrix vectorized acceleration)
    
    Supports three operating modes:
    - 'pretrain': Full pretraining augmentation mode (includes intensity jitter, m/z shift, 
                   residual peak removal, Rayleigh baseline and Gaussian noise clusters)
    - 'finetune_train': Lightweight fine-tuning augmentation mode (mild intensity jitter and 
                        baseline noise to prevent overfitting during fine-tuning)
    - 'eval': Evaluation/inference mode (identity mapping, no augmentation applied)
    """
    def __init__(self,
                 mode='pretrain',
                 # Pretraining intensity jitter and renormalization
                 pretrain_jitter_range=(0.6, 1.5),
                 mz_shift_range=1,
                 minor_peak_threshold_ratio=0.20,
                 minor_peak_removal_prob=0.30,
                 min_peaks_retain=3,
                 rayleigh_scale_tolerance=(0.005, 0.05),
                 n_clusters_range=(1, 5),
                 cluster_width_range=(3, 11),
                 cluster_amp_ratio_range=(0.10, 0.50),
                 # Fine-tuning lightweight augmentation parameters
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

        # Pre-compute lognormal parameters for Gaussian noise cluster sampling
        mean = (self.cluster_amp_ratio_range[0] + self.cluster_amp_ratio_range[1]) / 2.0
        sigma_val = (self.cluster_amp_ratio_range[1] - self.cluster_amp_ratio_range[0]) / 4.0
        self.log_mean = math.log(mean)
        self.log_sigma = sigma_val / mean
        
        # GPU static position tensor cache
        self._pos_cache = {}

    def __call__(self, spectrum: torch.Tensor) -> torch.Tensor:
        if self.mode == 'eval':
            return spectrum
            
        is_1d = (spectrum.dim() == 1)
        if is_1d:
            spec = spectrum.unsqueeze(0)
        else:
            spec = spectrum
            
        device = spec.device
        
        # 1. Restore to linear intensity space (square)
        spec = spec ** 2

        if self.mode == 'pretrain':
            # Dimension 1: Intensity jitter + renormalize to 999 (Sample-wise jitter, preserving topology)
            spec = self._intensity_jitter_with_renorm(spec, self.pretrain_jitter_range, device)
            # Dimension 2: m/z global shift (Vectorized gather)
            spec = self._mz_shift(spec, device)
            # Dimension 3: Probabilistic removal of low-abundance residual peaks (Vectorized mask)
            spec = self._remove_minor_peaks(spec, device)
            # Dimension 4: Rayleigh + Gaussian noise injection (Vectorized Batch noise)
            spec = self._inject_noise(spec, device)
        elif self.mode == 'finetune_train':
            # Fine-tuning lightweight augmentation: Sample-wise mild intensity jitter and weak baseline noise
            spec = self._intensity_jitter(spec, self.finetune_jitter_range, device)
            spec = self._inject_rayleigh_noise(spec, self.finetune_rayleigh_scale, device)
            
        # Re-apply TIC normalization + square root scaling
        res = self._re_normalize(spec)
        return res.squeeze(0) if is_1d else res

    def _intensity_jitter_with_renorm(self, spec: torch.Tensor, jitter_range: tuple, device: torch.device) -> torch.Tensor:
        """
        Sample-wise intensity jitter + renormalization to 999
        Multiplies each sample by a random scalar (B, 1), preserving relative ratios across different m/z peaks
        """
        B = spec.shape[0]
        jitter_factors = torch.empty((B, 1), device=device).uniform_(
            jitter_range[0], jitter_range[1]
        )
        spec = spec * jitter_factors
        
        spec_max = spec.max(dim=1, keepdim=True).values
        mask = spec_max > 1e-8
        spec = torch.where(mask, spec / (spec_max + 1e-12) * 999.0, spec)
        return spec

    def _intensity_jitter(self, spec: torch.Tensor, jitter_range: tuple, device: torch.device) -> torch.Tensor:
        """Sample-wise intensity jitter (B, 1)"""
        B = spec.shape[0]
        jitter_factors = torch.empty((B, 1), device=device).uniform_(
            jitter_range[0], jitter_range[1]
        )
        return spec * jitter_factors

    def _mz_shift(self, spec: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Fully vectorized m/z global shift of ±mz_shift_range
        Uses torch.gather to perform independent random shifts for each sample in the batch on GPU in one pass
        """
        if self.mz_shift_range <= 0:
            return spec
            
        B, N = spec.shape
        shifts = torch.randint(-self.mz_shift_range, self.mz_shift_range + 1, (B, 1), device=device)
        
        cols = torch.arange(N, device=device).unsqueeze(0)  # (1, N)
        src_cols = cols - shifts                            # (B, N)
        
        valid_mask = (src_cols >= 0) & (src_cols < N)
        src_cols_clamped = src_cols.clamp(0, N - 1)
        
        shifted_spec = torch.gather(spec, dim=1, index=src_cols_clamped)
        return torch.where(valid_mask, shifted_spec, 0.0)

    def _remove_minor_peaks(self, spec: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Probabilistic removal of low-abundance peaks (fully vectorized)
        1. Dynamic threshold = 20% of each sample's base peak intensity
        2. Each peak below threshold has 30% probability of removal
        3. Constraint: Ensure at least min_peaks_retain peaks remain per sample
        """
        B, N = spec.shape
        base_peaks = spec.max(dim=1, keepdim=True).values  # (B, 1)
        
        dynamic_thresholds = base_peaks * self.minor_peak_threshold_ratio  # (B, 1)
        minor_mask = (spec > 1e-8) & (spec < dynamic_thresholds)           # (B, N)
        
        rand_probs = torch.rand_like(spec)                                  # (B, N)
        candidate_removal = minor_mask & (rand_probs < self.minor_peak_removal_prob)
        
        n_total_before = (spec > 1e-8).sum(dim=1, keepdim=True)             # (B, 1)
        n_remove = candidate_removal.sum(dim=1, keepdim=True)              # (B, 1)
        
        valid_row_mask = (n_total_before - n_remove) >= self.min_peaks_retain # (B, 1)
        final_removal = candidate_removal & valid_row_mask
        
        return torch.where(final_removal, 0.0, spec)

    def _inject_rayleigh_noise(self, spec: torch.Tensor, scale_tolerance: tuple, device: torch.device) -> torch.Tensor:
        """Fully vectorized Rayleigh baseline noise injection"""
        B, N = spec.shape
        spec_max = spec.max(dim=1, keepdim=True).values
        mask = (spec_max > 1e-8)
        
        tic = spec.sum(dim=1, keepdim=True)                                # (B, 1)
        u = torch.empty((B, 1), device=device).uniform_(
            scale_tolerance[0], scale_tolerance[1]
        )
        rayleigh_scale = u * tic / float(N)                                # (B, 1)
        
        rayleigh_noise = torch.empty_like(spec).exponential_()             # (B, N)
        rayleigh_noise = torch.sqrt(rayleigh_noise) * rayleigh_scale
        
        noisy_spec = torch.clamp(spec + rayleigh_noise, min=0.0)
        return torch.where(mask, noisy_spec, spec)

    def _inject_noise(self, spec: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Differentiated noise injection: Rayleigh baseline noise + Gaussian noise clusters
        (Full Batch GPU parallelization with cache reuse)
        """
        B, N = spec.shape
        spec = self._inject_rayleigh_noise(spec, self.rayleigh_scale_tolerance, device)
        
        num_clusters = torch.randint(
            self.n_clusters_range[0], self.n_clusters_range[1] + 1, (B, 1), device=device
        )
        max_k = self.n_clusters_range[1]
        
        cache_key = (N, device.type, device.index if device.index is not None else 0)
        if cache_key not in self._pos_cache:
            self._pos_cache[cache_key] = torch.arange(N, device=device, dtype=torch.float32).unsqueeze(0)
        pos = self._pos_cache[cache_key]
        
        for k in range(max_k):
            active_mask = (num_clusters > k).float()                           # (B, 1)
            C = torch.randint(0, N, (B, 1), device=device)                      # (B, 1)
            W = torch.randint(
                self.cluster_width_range[0], self.cluster_width_range[1] + 1, (B, 1), device=device
            )                                                                   # (B, 1)
            
            dist = torch.abs(pos - C.float())                                   # (B, N)
            within_width = (dist <= (W.float() / 2.0)).float()                 # (B, N)
            
            sigma = W.float() / 6.0                                             # (B, 1)
            gaussian_shape = torch.exp(-0.5 * ((pos - C.float()) / (sigma + 1e-8)) ** 2) * within_width
            
            local_max = (spec * within_width).max(dim=1, keepdim=True).values   # (B, 1)
            amp_ratio = torch.empty((B, 1), device=device).log_normal_(
                mean=self.log_mean, std=self.log_sigma
            ).clamp(self.cluster_amp_ratio_range[0], self.cluster_amp_ratio_range[1])
            
            cluster_amp = local_max * amp_ratio                                 # (B, 1)
            gaussian_peak = cluster_amp * gaussian_shape * active_mask          # (B, N)
            spec = spec + gaussian_peak
            
        return torch.clamp(spec, min=0.0)

    def _re_normalize(self, spec: torch.Tensor) -> torch.Tensor:
        """TIC normalization + sqrt compression (fully vectorized)"""
        tic = spec.sum(dim=1, keepdim=True)
        mask = (tic > 1e-8)
        spec_norm = torch.where(mask, spec / (tic + 1e-12), spec)
        return torch.sqrt(torch.clamp(spec_norm, min=0.0))