"""
이미지 인코더 (Image Encoder) = Backbone + Neck

■ 역할:
  입력 이미지를 다중 스케일 특징맵으로 변환하는 전체 파이프라인.
  Backbone(Hiera)이 추출한 특징을 Neck(FPN)이 정제하여 출력합니다.

■ 구조:
  이미지 (B, 3, H, W)
    → Backbone (Hiera): 4개 스케일 특징맵 추출
    → Neck (FPN): 채널 통일 + Top-down 융합
    → 출력: vision_features, vision_pos_enc, backbone_fpn

■ FPN (Feature Pyramid Network):
  핵심 아이디어: 고해상도(세밀) + 저해상도(전체맥락) 정보를 융합
  
  저해상도 특징 ──> 업샘플링 ──> (+) 고해상도 특징
  
  ★ 산업 결함 검출에서 중요한 이유:
    - 작은 스크래치: 고해상도 레벨에서 감지
    - 넓은 영역 변색: 저해상도 레벨에서 감지
    - FPN이 두 정보를 융합 → 모든 크기의 결함 동시 감지
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """
    이미지 인코더 = Backbone(trunk) + Neck
    
    ■ scalp 파라미터:
      가장 낮은 해상도의 특징맵을 몇 개 버릴지 결정.
      예: scalp=1이면 가장 큰(최저해상도) 특징맵 1개를 제거.
      → SAM2 Tiny에서 scalp=1 사용 (효율성 위해)
    
    Args:
        trunk: Backbone 모듈 (Hiera)
        neck: FPN Neck 모듈
        scalp: 제거할 최저해상도 특징맵 수 (기본 0)
    """

    def __init__(
        self,
        trunk: nn.Module,
        neck: nn.Module,
        scalp: int = 0,
    ):
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        # Backbone과 Neck의 채널 수가 일치하는지 확인
        assert (
            self.trunk.channel_list == self.neck.backbone_channel_list
        ), (
            f"Channel dims of trunk and neck do not match. "
            f"Trunk: {self.trunk.channel_list}, Neck: {self.neck.backbone_channel_list}"
        )

    def forward(self, sample: torch.Tensor) -> dict:
        """
        Args:
            sample: (B, 3, H, W) 입력 이미지
        Returns:
            dict:
                - "vision_features": 최고 해상도 특징맵 (B, C, H', W')
                - "vision_pos_enc": 위치 인코딩 리스트
                - "backbone_fpn": FPN 출력 특징맵 리스트
        """
        # Backbone → Neck
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            # 최저해상도 특징맵 제거
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]
        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        return output


class FpnNeck(nn.Module):
    """
    FPN 넥 (Feature Pyramid Network Neck)
    
    ■ 역할:
      Backbone의 다중 스케일 특징맵을 받아서:
      1. 각 스케일의 채널 수를 d_model로 통일 (1×1 Conv)
      2. 저해상도 → 고해상도로 Top-down 정보 전파
      3. 각 스케일에 위치 인코딩 추가
    
    ■ 동작:
      Stage4 (최저해상도, 가장 풍부한 의미 정보)
        │ 1×1 Conv → d_model 채널
        │ ↓ (Top-down: 2배 업샘플링)
        ╰──(+)── Stage3 (1×1 Conv)
            │ ↓ (Top-down: 2배 업샘플링) 
            ╰──(+)── Stage2 (1×1 Conv)
                │
                Stage1 (1×1 Conv) ← Top-down 없이 Lateral만
      
      ★ fpn_top_down_levels로 어떤 레벨에 Top-down을 적용할지 제어
        Tiny 기준: [2, 3] → Level 2, 3에만 Top-down 적용
    
    ■ 비유:
      부장(Stage4, 전체 맥락)의 판단이 → 과장(Stage3) → 대리(Stage2)로
      전달되어 세부 업무(세밀한 특징)에 반영되는 구조
    
    Args:
        position_encoding: 위치 인코딩 모듈
        d_model: 출력 통일 차원 (보통 256)
        backbone_channel_list: 각 Stage의 입력 채널 수 리스트
        fpn_interp_model: 업샘플링 보간 방식 ("nearest" 또는 "bilinear")
        fpn_top_down_levels: Top-down 정보를 전파할 레벨 인덱스
    """

    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ):
        super().__init__()
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        self.d_model = d_model

        # 각 스케일용 1×1 Conv (채널 통일)
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )
            self.convs.append(current)

        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type

        # Top-down을 적용할 레벨 설정
        if fpn_top_down_levels is None:
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]):
        """
        Args:
            xs: Backbone 출력 — 다중 스케일 특징맵 리스트
                [Stage4_feat, Stage3_feat, Stage2_feat, Stage1_feat]
        Returns:
            out: FPN 출력 특징맵 리스트 (모두 d_model 채널)
            pos: 각 특징맵의 위치 인코딩 리스트
        """
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)

        # Top-down 순서: 저해상도(인덱스 큰 쪽) → 고해상도(인덱스 작은 쪽)
        prev_features = None
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)

            if i in self.fpn_top_down_levels and prev_features is not None:
                # Top-down: 이전(저해상도) 특징을 2배 업샘플 + 현재 특징과 합산
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(
                        None if self.fpn_interp_model == "nearest" else False
                    ),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features

            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos
