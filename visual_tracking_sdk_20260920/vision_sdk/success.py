"""Pure template matcher; does not click, access accounts or import host code."""
from pathlib import Path
import cv2
import numpy as np
from liescc import locate

SCALES=(.40,.45,.50,.52,.54,.55,.56,.57,.58,.60,.62,.65,.70,.80,.90,1.,1.1,1.2,1.3)

class SuccessDetector:
    def __init__(self,region='CN'):
        if region not in ('CN','TW'):raise ValueError('region must be CN or TW')
        root=Path(locate.__file__).parent
        path=root/('tw/tw-cg.png' if region=='TW' else 'cg.png')
        self.template=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
        if self.template is None:raise ValueError('Missing success template')
        self.reset()

    def reset(self):self._last=float('-inf');self._hits=0;self.status='NONE'

    def update(self,frame,timestamp,roi):
        if timestamp-self._last<.5:return self.status
        self._last=timestamp
        x,y,w,h=roi
        ow,oh=w/locate.HOLE_RW,h/locate.HOLE_RH
        ox,oy=x-ow*locate.HOLE_RX,y-oh*locate.HOLE_RY
        x0=max(0,int(ox-ow*.12));y0=max(0,int(oy-oh*.12))
        x1=min(frame.shape[1],int(ox+ow*1.12));y1=min(frame.shape[0],int(oy+oh*1.12))
        gray=cv2.cvtColor(frame[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
        best=-1.
        for scale in SCALES:
            tw,th=int(self.template.shape[1]*scale),int(self.template.shape[0]*scale)
            if tw<16 or th<12 or tw>gray.shape[1] or th>gray.shape[0]:continue
            templ=cv2.resize(self.template,(tw,th),interpolation=cv2.INTER_AREA)
            score=cv2.minMaxLoc(cv2.matchTemplate(gray,templ,cv2.TM_CCOEFF_NORMED))[1]
            best=max(best,score)
        self._hits=self._hits+1 if best>=.86 else 0
        self.status='SUCCESS' if self._hits>=2 else ('SUCCESS_PENDING' if self._hits else 'NONE')
        return self.status
