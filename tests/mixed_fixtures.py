"""Generated geometry fixtures: held apart from conference-specific image labels."""
import cv2
import numpy as np


def make_slide(width=1920,height=1080,title='Mixed layout validation',number='47.8'):
    im=np.full((height,width,3),248,np.uint8)
    factor=width/1920
    def text(s,x,y,size=1.3):
        cv2.putText(im,s,(round(x*factor),round(y*height/1080)),cv2.FONT_HERSHEY_SIMPLEX,size*factor,(90,40,10),max(1,round(2*factor)))
    text(title,70,95,1.8)
    for j,s in enumerate(['Treatment outcomes in the conference study','High resolution text must be preserved',
                          'Cohort A and cohort B: primary analysis','Follow-up duration: 24 months',
                          'Study result: '+number+' percent','Summary of the trial population']):
        text(s,70,220+j*95,1.3)
    cv2.rectangle(im,(round(width*.06),round(height*.79)),(round(width*.92),round(height*.92)),(160,70,30),max(2,round(3*factor)))
    text('Do not confuse 47.8 with 42.8',120,880,1.2)
    return im


def embed(slide,size,rect,background=(100,65,45)):
    w,h=size; x,y,rw,rh=rect
    frame=np.full((h,w,3),background,np.uint8)
    frame[y:y+rh,x:x+rw]=cv2.resize(slide,(rw,rh),interpolation=cv2.INTER_AREA)
    return frame


def fixtures():
    wide=make_slide()
    standard=make_slide(1440,1080,'Four by three slide')
    portrait=make_slide(900,1600,'Portrait slide')
    cases=[('wide-full',wide,(0,0,1,1)),
           ('left',embed(wide,(2560,1440),(40,220,1700,956)),(40/2560,220/1440,1740/2560,1176/1440)),
           ('right',embed(wide,(2560,1440),(800,180,1600,900)),(800/2560,180/1440,2400/2560,1080/1440)),
           ('browser',embed(wide,(2300,1500),(250,200,1800,1012),(65,65,65)),(250/2300,200/1500,2050/2300,1212/1500)),
           ('small',embed(wide,(1920,1080),(800,400,960,540)),(800/1920,400/1080,1760/1920,940/1080)),
           ('four-three-full',standard,(0,0,1,1)),
           ('four-three-right',embed(standard,(2400,1440),(900,180,1440,1080)),(900/2400,180/1440,2340/2400,1260/1440)),
           ('portrait-full',portrait,(0,0,1,1)),
           ('portrait-inset',embed(portrait,(1920,2160),(500,220,900,1600)),(500/1920,220/2160,1400/1920,1820/2160))]
    quad=np.float32([[270,250],[2090,180],[2010,1250],[340,1300]])
    transform=cv2.getPerspectiveTransform(np.float32([[0,0],[1919,0],[1919,1079],[0,1079]]),quad)
    warped=cv2.warpPerspective(wide,transform,(2560,1440),borderValue=(50,35,25))
    cases.append(('perspective',warped,None))
    cases.append(('changed-number',make_slide(number='42.8'),(0,0,1,1)))
    return cases
