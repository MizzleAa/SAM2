"""
Test 09: Video Predictor 검증

 목표:
  1. VideoPredictor 초기화
  2. init_state + add_points + propagate 패턴 검증
  3. 메모리 기반 추적 동작 확인
  4. 다중 객체 추적
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import pytest

from hvs.predictor.video_predictor import VideoPredictor

CKPT_PATH = r"c:\workspace\SAM2\facebook\checkpoints\sam2.1_hiera_tiny.pt"


def _make_synthetic_video(num_frames=5, h=256, w=256):
    """합성 비디오 생성 (이동하는 원)"""
    frames = []
    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cx = 80 + i * 20
        cy = 128
        yy, xx = np.ogrid[:h, :w]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < 30 ** 2
        frame[mask] = [200, 100, 50]
        frames.append(frame)
    return frames


class TestVideoPredictor:
    """기본 동작 검증"""

    def test_init(self):
        """VideoPredictor 초기화"""
        vp = VideoPredictor(
            model_size="tiny", image_size=256,
            device="cpu", init_mode="scratch",
        )
        assert vp.ie is not None
        assert vp.mem_enc is not None
        assert vp.mem_attn is not None
        print(f"  [OK] VideoPredictor init")

    def test_init_state(self):
        """init_state: 프레임 로드"""
        vp = VideoPredictor(
            model_size="tiny", image_size=256,
            device="cpu", init_mode="scratch",
        )
        frames = _make_synthetic_video(5)
        state = vp.init_state(frames)

        assert state["num_frames"] == 5
        assert state["orig_hw"] == (256, 256)
        assert 0 in state["cached_features"]  # 첫 프레임 캐시
        print(f"  [OK] init_state ({state['num_frames']} frames)")

    def test_add_points(self):
        """add_points: 첫 프레임에 프롬프트 추가"""
        vp = VideoPredictor(
            model_size="tiny", image_size=256,
            device="cpu", init_mode="scratch",
        )
        frames = _make_synthetic_video(5)
        state = vp.init_state(frames)

        frame_idx, obj_ids, masks = vp.add_points(
            state, frame_idx=0, obj_id=1,
            points=np.array([[80, 128]]),
            labels=np.array([1]),
        )

        assert frame_idx == 0
        assert obj_ids == [1]
        assert masks.shape == (1, 256, 256), f"Expected (1, 256, 256), got {masks.shape}"
        print(f"  [OK] add_points: masks {masks.shape}")

    def test_propagate(self):
        """propagate: 전체 프레임 전파"""
        vp = VideoPredictor(
            model_size="tiny", image_size=256,
            device="cpu", init_mode="scratch",
        )
        frames = _make_synthetic_video(5)
        state = vp.init_state(frames)

        # 첫 프레임에 점 추가
        vp.add_points(
            state, frame_idx=0, obj_id=1,
            points=np.array([[80, 128]]),
            labels=np.array([1]),
        )

        # 전파
        results = []
        for frame_idx, obj_ids, masks in vp.propagate(state):
            results.append((frame_idx, masks.shape))

        assert len(results) == 5
        for idx, (fi, shape) in enumerate(results):
            assert fi == idx
            assert shape == (1, 256, 256)
        print(f"  [OK] propagate: {len(results)} frames tracked")

    def test_multi_object(self):
        """다중 객체 추적"""
        vp = VideoPredictor(
            model_size="tiny", image_size=256,
            device="cpu", init_mode="scratch",
        )
        frames = _make_synthetic_video(3)
        state = vp.init_state(frames)

        # 두 개 객체
        vp.add_points(state, 0, obj_id=1, points=np.array([[80, 128]]), labels=np.array([1]))
        vp.add_points(state, 0, obj_id=2, points=np.array([[180, 128]]), labels=np.array([1]))

        results = list(vp.propagate(state))
        assert len(results) == 3
        # 각 프레임에서 2개 객체 마스크
        for fi, obj_ids, masks in results:
            assert len(obj_ids) == 2
            assert masks.shape[0] == 2  # 2 objects
        print(f"  [OK] Multi-object: 2 objects, {len(results)} frames")


class TestVideoPredictorCheckpoint:
    """실제 체크포인트 검증"""

    def test_checkpoint_propagate(self):
        """실제 체크포인트로 전파"""
        if not os.path.exists(CKPT_PATH):
            pytest.skip("Checkpoint not found")

        vp = VideoPredictor(
            model_size="tiny", image_size=256,
            device="cpu", checkpoint_path=CKPT_PATH,
            init_mode="finetune",
        )
        frames = _make_synthetic_video(3, h=256, w=256)
        state = vp.init_state(frames)

        vp.add_points(state, 0, obj_id=1, points=np.array([[80, 128]]), labels=np.array([1]))

        results = list(vp.propagate(state))
        assert len(results) == 3
        print(f"  [OK] Checkpoint propagate: {len(results)} frames")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
