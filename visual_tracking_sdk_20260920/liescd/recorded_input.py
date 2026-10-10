"""Read and verify the exact image stream accepted by a tracking session."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


class RecordedInput:
    def __init__(self, video_path):
        self.frames: dict[int, dict] = {}
        self.summary = None
        path = Path(video_path).with_suffix('.frames.jsonl')
        if not path.is_file():
            return
        for line in path.read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            if item.get('event') == 'recording_summary':
                self.summary = item
            elif 'frame' in item and 'timestamp_s' in item:
                self.frames[int(item['frame'])] = item

    @property
    def exact(self):
        return any(row.get('recording_format') == 'tracking-input-v1' for row in self.frames.values())

    def validate_complete(self, frame_count: int):
        if not self.exact:
            return
        rows = [self.frames.get(i, {}) for i in range(frame_count)]
        if (not self.summary or self.summary.get('error') or self.summary.get('dropped_frames')
                or self.summary.get('submitted_frames') != frame_count
                or self.summary.get('written_frames') != frame_count
                or len(self.frames) != frame_count
                or any(row.get('submitted_index') != i or not row.get('settings')
                       or 'run_yolo' not in row or not math.isfinite(row.get('timestamp_s', float('nan')))
                       or not (row.get('host_result') or row.get('local_result'))
                       or ('host_result' in row and row['host_result'].get('host_frame_index') != i)
                       for i, row in enumerate(rows))):
            raise ValueError('录像缺少追踪输入帧，无法验证原始路径；可取消“按录像参数回放”作普通分析')
        timestamps = [row['timestamp_s'] for row in rows]
        if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError('录像追踪时间未严格递增，无法验证原始路径')

    def image(self, index: int, frame: np.ndarray) -> np.ndarray:
        row = self.frames.get(index, {})
        if row.get('recording_format') not in {'tracking-input-v1', 'tracking-preview-v1'}:
            return frame
        w, h = int(row['content_width']), int(row['content_height'])
        if not (0 < w <= frame.shape[1] and 0 < h <= frame.shape[0]):
            raise ValueError(f'录像第{index}帧尺寸无效')
        image = np.ascontiguousarray(frame[:h, :w])
        if (row.get('recording_format') == 'tracking-input-v1'
                and hashlib.sha256(image.tobytes()).hexdigest() != row['image_sha256']):
            raise ValueError(f'录像第{index}帧像素校验失败')
        return image

    def compare(self, index, tracked, smoothed):
        row = self.frames.get(index, {})
        if row.get('recording_format') == 'tracking-preview-v1':
            return None
        remote = row.get('host_result')
        reference = remote or row.get('local_result')
        if not reference:
            return None
        distance = 0.0
        mismatch = (reference.get('target_id') != tracked.target_id
                    or reference.get('lock_state') != tracked.state.value)
        for expected, actual in ((reference.get('raw_aim_point'), tracked.aim_point),
                                 (reference.get('aim_point'), smoothed)):
            if expected is None or actual is None:
                mismatch |= (expected is None) != (actual is None)
                continue
            expected = np.asarray(expected, dtype=float)
            if remote is not None:
                expected *= (row['content_width'], row['content_height'])
            distance = max(distance, float(np.linalg.norm(expected - actual)))
        return bool(mismatch or distance > 1e-6), distance
