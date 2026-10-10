import unittest
from unittest.mock import patch
import numpy as np
from vision_sdk import Session,MouseOutput,Observation
from vision_sdk.success import SuccessDetector
from liescd.detector import Detection
from liescd.tracker import TrackResult,TrackState

class FakeDetector:
    def detect(self,frame,confidence):self.shape=frame.shape;return []

class FakeTracker:
    def set_frame_size(self,*size):pass
    def update(self,*args):return TrackResult(TrackState.LOCKED,(100.,50.),None,True,1)

class Tests(unittest.TestCase):
    def test_coordinate_and_reset(self):
        s=Session(detector=FakeDetector(),success_detection=False)
        s.reset(roi=(100,50,1280,720));s.tracker=FakeTracker()
        out=s.process(np.zeros((900,1500,3),np.uint8),1.,screen_origin=(-200,30))
        self.assertEqual(out.aim_frame,(300,150));self.assertEqual(out.aim_screen,(100,180))
        self.assertEqual(s.detector.shape,(360,640,3))
        s.reset(roi=(0,0,640,480));self.assertEqual(len(s.trail),0)
        out=s.process(np.zeros((480,640,3),np.uint8),0.)
        self.assertIsNone(out.aim_screen);s.close();s.close()
        with self.assertRaises(RuntimeError):s.process(np.zeros((480,640,3),np.uint8),2.)

    def test_invalid_and_geometry(self):
        with Session(detector=FakeDetector(),success_detection=False) as s:
            s.reset(roi=(0,0,640,480));f=np.zeros((480,640,3),np.uint8)
            s.process(f,1.)
            with self.assertRaises(ValueError):s.process(f,1.)
            with self.assertRaises(ValueError):s.process(f,float('nan'))
            self.assertEqual(s.process(f,2.,screen_origin=(1,0)).phase,'ENDED')
            self.assertEqual(s.process(f,3.).phase,'ENDED')

    def test_mouse(self):
        calls=[];m=MouseOutput(lambda x,y:calls.append((x,y)))
        def out(t,phase='LOCKED'):return Observation(t,phase,aim_screen=(11.,12.),session_id=1)
        self.assertFalse(m.emit(out(1.),now=1.3))
        m.manual(True,2.);self.assertFalse(m.emit(out(2.),now=2.01))
        m.manual(False,2.1);self.assertFalse(m.emit(out(2.2),now=2.2))
        self.assertTrue(m.emit(out(2.3),now=2.31))
        self.assertFalse(m.emit(out(2.3),now=2.32))
        self.assertFalse(m.emit(out(2.4,'LOST'),now=2.41))
        self.assertFalse(m.emit(out(2.5,'SUCCESS_PENDING'),now=2.51))
        self.assertFalse(m.emit(out(2.6),now=2.61))
        self.assertEqual(calls,[(11,12)])

    def test_success_template(self):
        d=SuccessDetector();t=d.template
        frame=np.zeros((t.shape[0]+40,t.shape[1]+40,3),np.uint8)
        frame[20:20+t.shape[0],20:20+t.shape[1]]=t[:,:,None]
        roi=(0,0,frame.shape[1],frame.shape[0])
        self.assertEqual(d.update(frame,0.,roi),'SUCCESS_PENDING')
        self.assertEqual(d.update(frame,.1,roi),'SUCCESS_PENDING')
        self.assertEqual(d.update(frame,.5,roi),'SUCCESS')
        d.reset();self.assertEqual(d.update(np.zeros_like(frame),1.,roi),'NONE')

    def test_locate_consensus(self):
        from liescc.locate import LieWindowCandidate
        c=LieWindowCandidate((0,0,100,100),(10,10,64,64),.99,.99,1.)
        with Session(detector=FakeDetector(),success_detection=False) as s:
            with patch('vision_sdk.session.locate.locate_by_dual_anchor',return_value=c):
                f=np.zeros((100,100,3),np.uint8)
                self.assertEqual(s.process(f,0.).phase,'LOCATING')
                self.assertEqual(s.process(f,1.).phase,'LOCATING')
                self.assertNotEqual(s.process(f,2.).phase,'LOCATING')
                self.assertEqual(s.roi,(10,10,64,64))

    def test_real_locator_template(self):
        from liescc import locate
        from pathlib import Path
        import cv2
        image=cv2.imdecode(np.fromfile(Path(locate.__file__).parent/'startlies2.png',np.uint8),cv2.IMREAD_COLOR)
        for scale in (.4,.6,1.):
            resized=cv2.resize(image,None,fx=scale,fy=scale)
            canvas=np.full((resized.shape[0]+200,resized.shape[1]+200,3),30,np.uint8)
            canvas[100:100+resized.shape[0],100:100+resized.shape[1]]=resized
            candidate=locate.locate_by_dual_anchor(canvas)
            self.assertIsNotNone(candidate)
            self.assertLessEqual(abs(candidate.outer_rect[0]-100),2)

    def test_success_stops_output(self):
        with Session(detector=FakeDetector()) as s:
            s.reset(roi=(0,0,64,64));s.tracker=FakeTracker();f=np.zeros((64,64,3),np.uint8)
            self.assertIsNotNone(s.process(f,0.).result)
            with patch.object(s.success,'update',return_value='SUCCESS'):
                out=s.process(f,16.)
                self.assertEqual(out.phase,'SUCCESS');self.assertIsNone(out.aim_screen)
                self.assertEqual(s.process(f,17.).phase,'SUCCESS')
            s.reset(roi=(0,0,64,64));self.assertNotEqual(s.process(f,0.).phase,'SUCCESS')

    def test_timeout_and_error(self):
        with Session(detector=FakeDetector(),success_detection=False,max_duration_s=1.) as s:
            s.reset(roi=(0,0,64,64));f=np.zeros((64,64,3),np.uint8)
            s.process(f,0.);self.assertEqual(s.process(f,2.).phase,'ENDED')
        with Session(detector=FakeDetector(),success_detection=False) as s:
            s.reset(roi=(0,0,64,64))
            with patch.object(s.detector,'detect',side_effect=RuntimeError('test')):
                with self.assertRaises(RuntimeError):s.process(f,0.)
            self.assertEqual(s.process(f,1.).phase,'ENDED')

if __name__=='__main__':unittest.main()
