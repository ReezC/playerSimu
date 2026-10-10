"""python -m examples.video path/to/content_video.mp4 [--full-frame]"""
import argparse
import cv2
from vision_sdk import Session
from liescd.video_io import open_video_capture
from liescd.gui import render_frame

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('video');p.add_argument('--full-frame',action='store_true')
    args=p.parse_args();cap=open_video_capture(args.video)
    if not cap.isOpened():raise RuntimeError('Cannot open video')
    try:
        with Session(success_detection=args.full_frame) as session:
            session.prepare();i=0;last=None
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.
            while True:
                ok,frame=cap.read()
                if not ok:break
                if i==0 and not args.full_frame:session.reset(roi=(0,0,frame.shape[1],frame.shape[0]))
                t=getattr(cap,'timestamp_s',None)
                if t is None:t=i/fps
                # Reject duplicate/nonmonotonic timestamps, do not invent observations.
                if last is not None and t<=last:raise ValueError('Nonmonotonic video timestamps')
                last=t
                out=session.process(frame,t)
                visual=frame if out.result is None else render_frame(out.tracking_frame,out.detections,out.result,out.trail,True)
                cv2.imshow('Tracking SDK - Q to stop',visual)
                if cv2.waitKey(1)&255==ord('q') or out.phase in ('SUCCESS','ENDED'):break
                i+=1
    finally:cap.release();cv2.destroyAllWindows()

if __name__=='__main__':main()
