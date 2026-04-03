"""
SAM2 VideoPredictor 유틸리티

■ select_closest_cond_frames():
  시간적으로 가장 가까운 조건 프레임을 선택합니다.
  Memory Attention에서 너무 많은 조건 프레임이 있을 때
  가장 관련성 높은 N개만 선택하여 GPU 메모리를 절약.

■ get_1d_sine_pe():
  1D 사인 위치 인코딩. Object Pointer의 시간적 위치를 인코딩할 때 사용.
  각 프레임 간 거리를 연속적인 사인/코사인 벡터로 변환.
"""

import torch
import torch.nn.functional as F


def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frames_in_attn):
    """
    시간적으로 가장 가까운 조건 프레임 선택

    ■ 동작:
      조건 프레임이 max_cond_frames_in_attn보다 많으면,
      현재 frame_idx와 가장 가까운 N개만 선택합니다.
      나머지는 unselected로 반환 (비조건 메모리로 활용 가능).

    Args:
        frame_idx: 현재 프레임 인덱스
        cond_frame_outputs: {frame_idx: output} 조건 프레임 출력 딕셔너리
        max_cond_frames_in_attn: 최대 사용할 조건 프레임 수 (-1이면 전부)

    Returns:
        (selected_outputs, unselected_outputs): 선택된/비선택된 딕셔너리
    """
    if max_cond_frames_in_attn == -1 or len(cond_frame_outputs) <= max_cond_frames_in_attn:
        selected = cond_frame_outputs
        unselected = {}
    else:
        # 현재 프레임과의 거리로 정렬
        assert max_cond_frames_in_attn >= 2, "need at least 2 cond frames"
        t_diff = [abs(frame_idx - t) for t in cond_frame_outputs]
        sorted_indices = sorted(range(len(t_diff)), key=lambda i: t_diff[i])
        cond_keys = list(cond_frame_outputs.keys())

        selected_keys = set(cond_keys[i] for i in sorted_indices[:max_cond_frames_in_attn])
        selected = {k: v for k, v in cond_frame_outputs.items() if k in selected_keys}
        unselected = {k: v for k, v in cond_frame_outputs.items() if k not in selected_keys}

    return selected, unselected


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """
    1D 사인 위치 인코딩

    ■ Object Pointer의 시간적 위치를 인코딩할 때 사용.
      현재 프레임과의 시간 거리를 연속적인 벡터로 변환합니다.

    Args:
        pos_inds: (N,) 위치 인덱스 (정규화된 0~1 범위)
        dim: 출력 차원
        temperature: PE 스케일 파라미터

    Returns:
        (N, dim) 위치 인코딩 벡터
    """
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos = pos_inds.float().unsqueeze(-1) / dim_t
    pos = torch.cat([pos.sin(), pos.cos()], dim=-1)

    # dim이 홀수인 경우 패딩
    if dim % 2 != 0:
        pos = F.pad(pos, (0, 1))

    return pos
