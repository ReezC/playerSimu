"""Screen capture adapter. Preview by default; --move explicitly enables SetCursorPos."""
import argparse
import ctypes
import os
import time
import cv2
import numpy as np
import mss
from vision_sdk import Session,MouseOutput
from liescd.gui import render_frame

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rect',nargs=4,type=int,required=True,metavar=('LEFT','TOP','WIDTH','HEIGHT'))
    p.add_argument('--content',action='store_true',help='Rectangle is already the content ROI')
    p.add_argument('--move',action='store_true',help='Actually move the Windows cursor')
    p.add_argument('--region',choices=('CN','TW'),default='CN')
    a=p.parse_args()
    if os.name!='nt':raise RuntimeError('Live example requires Windows')
    user32=ctypes.windll.user32
    try:ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError,OSError):user32.SetProcessDPIAware()
    left,top,width,height=a.rect
    if min(width,height)<32:raise ValueError('Capture dimensions must be at least 32')
    box=dict(left=left,top=top,width=width,height=height)
    def move(x,y):
        # Recheck keys directly before emitting OS input.
        if user32.GetAsyncKeyState(0x01)&0x8000 or user32.GetAsyncKeyState(0x1B)&0x8000:return
        if not user32.SetCursorPos(x,y):raise OSError('SetCursorPos failed')
    output=MouseOutput(move)
    try:
        with mss.mss() as capture,Session(region=a.region,success_detection=not a.content) as session:
            session.prepare()
            if a.content:session.reset(roi=(0,0,width,height))
            while not user32.GetAsyncKeyState(0x1B)&0x8000:
                started=time.monotonic()
                frame=np.asarray(capture.grab(box))[:,:,:3].copy()
                output.manual(bool(user32.GetAsyncKeyState(0x01)&0x8000))
                out=session.process(frame,started,screen_origin=(left,top))
                output.manual(bool(user32.GetAsyncKeyState(0x01)&0x8000))
                if a.move:output.emit(out)
                image=frame if out.result is None else render_frame(out.tracking_frame,out.detections,out.result,out.trail,True)
                cv2.imshow('Tracking preview - ESC stops; keep outside capture area',image)
                if cv2.waitKey(1)&255==27 or out.phase in ('SUCCESS','ENDED'):break
                time.sleep(max(0.,.05-(time.monotonic()-started)))
    finally:cv2.destroyAllWindows()

if __name__=='__main__':main()
