from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import math
import time

import cv2
import numpy as np

from liescc import locate
from liescd.detector import GeometryDetector, DEFAULT_WEIGHTS
from liescd.ocsort_tracker import create_liescd_ocsort_backend, OCSORT_LOW_CONFIDENCE
from liescd.runtime_tracker import LiescdTargetTracker, LIESCD_TRACK_BUFFER
from liescd.realtime_stream import prepare_tracking_frame
from liescd.trajectory import AimTrajectorySmoother
from .success import SuccessDetector


@dataclass(frozen=True)
class Observation:
    """aim_frame/aim_screen use original pixels; result/detections use tracking pixels."""
    timestamp_s: float
    phase: str
    roi: tuple | None = None
    aim_frame: tuple | None = None
    aim_screen: tuple | None = None
    result: object = None
    tracking_frame: object = None
    detections: tuple = ()
    trail: tuple = ()
    reason: str = ''
    session_id: int = 0


class Session:
    """Single-owner synchronous API. One session per task, one BGR frame per update.

    Create/prepare before the task. Source timestamps must strictly increase.
    Full-frame mode discovers and freezes ROI; roi=... bypasses discovery.
    Context manager closes only its own detector. No OS mouse calls are made.
    """
    def __init__(self, *, weights=DEFAULT_WEIGHTS, confidence=.15, detector=None,
                 success_detection=True, region='CN', max_duration_s=120.):
        if not 0.01 <= confidence <= 0.99:
            raise ValueError('confidence must be in [0.01, 0.99]')
        if not math.isfinite(max_duration_s) or max_duration_s <= 0:
            raise ValueError('max_duration_s must be positive')
        self.confidence=float(confidence)
        self.max_duration_s=float(max_duration_s)
        self._owns_detector=detector is None
        self.detector=detector if detector is not None else GeometryDetector(weights)
        self.success=SuccessDetector(region) if success_detection else None
        self.closed=False
        self.session_id=0
        self.reset()

    def prepare(self):
        if self.closed:raise RuntimeError('Session is closed')
        if not self.detector.prepare_async().result():
            raise RuntimeError(self.detector.error)

    def reset(self, *, roi=None):
        if getattr(self,'closed',False):raise RuntimeError('Session is closed')
        self.session_id+=1
        self.roi=tuple(roi) if roi is not None else None
        self._shape=None;self._origin=None;self._previous_time=None;self._start_time=None
        self._roi_time=None;self._candidates=deque(maxlen=20)
        self.tracker=LiescdTargetTracker(backend=create_liescd_ocsort_backend(
            track_buffer=LIESCD_TRACK_BUFFER,detection_threshold=self.confidence))
        self.smoother=AimTrajectorySmoother()
        self.trail=deque(maxlen=2000)
        self.phase='LOCATING' if roi is None else 'TRACKING'
        self._terminal=False
        if self.success:self.success.reset()

    def finish(self, reason='stopped'):
        self._terminal=True;self.phase='ENDED';self.trail.clear()
        return Observation(self._previous_time or 0.,self.phase,self.roi,reason=reason,session_id=self.session_id)

    def process(self, frame_bgr, timestamp_s, *, screen_origin=(0,0)):
        if self.closed:raise RuntimeError('Session is closed')
        t=float(timestamp_s)
        if not math.isfinite(t) or (self._previous_time is not None and t<=self._previous_time):
            raise ValueError('Use strictly increasing finite source timestamps')
        if not isinstance(frame_bgr,np.ndarray) or frame_bgr.dtype!=np.uint8 or frame_bgr.ndim!=3 or frame_bgr.shape[2]!=3:
            raise ValueError('Expected H x W x 3 uint8 BGR image')
        if min(frame_bgr.shape[:2])<32:raise ValueError('Frame too small')
        origin=tuple(float(v) for v in screen_origin)
        if len(origin)!=2 or not all(math.isfinite(v) for v in origin):raise ValueError('Invalid origin')
        self._previous_time=t
        if self._terminal:
            return Observation(t,self.phase,self.roi,reason='Call reset for a new task',session_id=self.session_id)
        if self._start_time is None:self._start_time=t
        if t-self._start_time>self.max_duration_s:return self.finish('timeout')
        if self._shape is not None and (self._shape!=frame_bgr.shape or self._origin!=origin):
            return self.finish('capture_geometry_changed; reset with new ROI')
        self._shape=frame_bgr.shape;self._origin=origin
        if self.roi is None:
            candidate=locate.locate_by_dual_anchor(frame_bgr)
            self._candidates.append(candidate)
            if candidate is None or sum(locate.candidates_are_stable(c,candidate) for c in self._candidates)<3:
                return Observation(t,'LOCATING',session_id=self.session_id)
            self.roi=candidate.content_rect
        if self._roi_time is None:self._roi_time=t
        if len(self.roi)!=4 or any(not isinstance(v,(int,np.integer)) for v in self.roi):
            raise ValueError('ROI must be integer (x,y,width,height)')
        x,y,w,h=self.roi
        if min(x,y)<0 or min(w,h)<32 or x+w>frame_bgr.shape[1] or y+h>frame_bgr.shape[0]:
            raise ValueError('ROI outside image or smaller than 32 pixels')
        # Stop output on the FIRST panel observation; confirm success on two scans.
        if self.success and t-self._roi_time>=15.:
            status=self.success.update(frame_bgr,t,self.roi)
            if status!='NONE':
                self.phase=status
                if status=='SUCCESS':self._terminal=True
                return Observation(t,status,self.roi,reason='Host owns confirmation click',session_id=self.session_id)
        tracking,_=prepare_tracking_frame(frame_bgr[y:y+h,x:x+w])
        self.tracker.set_frame_size(tracking.shape[1],tracking.shape[0])
        try:
            detections=self.detector.detect(tracking,min(self.confidence,OCSORT_LOW_CONFIDENCE))
            result=self.tracker.update(t,detections)
            point=self.smoother.update(t,result.aim_point,result.target_bbox)
        except Exception:
            self.finish('processing_error')
            raise
        from dataclasses import replace
        result=replace(result,aim_point=point)
        if result.trail_break:self.trail.append(None)
        if point is not None:self.trail.append(point)
        if point is not None and not all(math.isfinite(v) for v in point):
            return self.finish('nonfinite_output')
        aim=None if point is None else (x+point[0]*w/tracking.shape[1],y+point[1]*h/tracking.shape[0])
        # Suppress predictions outside the input ROI instead of moving outside it.
        if aim is not None and not (x<=aim[0]<x+w and y<=aim[1]<y+h):aim=None
        screen=None if aim is None else (origin[0]+aim[0],origin[1]+aim[1])
        self.phase=result.state.value
        return Observation(t,self.phase,self.roi,aim,screen,result,tracking,
            tuple(d for d in detections if d.confidence>=self.confidence),tuple(self.trail),session_id=self.session_id)

    def close(self):
        if not self.closed:
            self.finish('closed');self.closed=True
            if self._owns_detector:self.detector.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()


class MouseOutput:
    """Optional callback adapter. Never queues points; caller polls manual hold.

    All times must share time.monotonic's clock. No automatic clicks/interpolation.
    A new task requires reset, including after SUCCESS_PENDING.
    """
    def __init__(self, move, *, max_age_s=.2, release_grace_s=.15):
        if not math.isfinite(max_age_s) or max_age_s<=0:raise ValueError('Invalid max_age_s')
        if not math.isfinite(release_grace_s) or release_grace_s<0:raise ValueError('Invalid release_grace_s')
        self.move=move;self.max_age_s=max_age_s;self.release_grace_s=release_grace_s
        self.reset()

    def reset(self):
        self.blocked=False;self._held=False;self._release_at=None;self._seen=float('-inf');self._session_id=None

    def manual(self, held, now=None):
        now=time.monotonic() if now is None else now
        if self._held and not held:self._release_at=now
        self._held=bool(held)

    def emit(self, observation, *, now=None):
        now=time.monotonic() if now is None else float(now)
        if self._session_id is None:self._session_id=observation.session_id
        if observation.session_id!=self._session_id:return False
        if observation.phase in ('SUCCESS_PENDING','SUCCESS','ENDED'):self.blocked=True
        if observation.timestamp_s<=self._seen:return False
        self._seen=observation.timestamp_s
        if self.blocked or self._held or (self._release_at is not None and now-self._release_at<self.release_grace_s):return False
        age=now-observation.timestamp_s
        if not math.isfinite(age) or not 0<=age<=self.max_age_s:return False
        if observation.phase not in ('LOCKED','COAST') or observation.aim_screen is None:return False
        x,y=observation.aim_screen
        if not all(math.isfinite(v) for v in (x,y)):return False
        self.move(round(x),round(y));return True
