import numpy as np
import cv2
import math

class Turtle:
    def __init__(self,pos=(0,0),dir=(0,1)):
        self.pos = pos
        self.dir = dir
        self.stack_pos = []
        self.stack_dir = []
    def go(self,length):
        self.pos = (
            self.pos[0] + self.dir[0]*length,
            self.pos[1] + self.dir[1]*length
        )
    def left(self,angle):
        angle *= math.pi/180
        self.dir = (
            self.dir[0]*math.cos(angle)-self.dir[1]*math.sin(angle), 
            self.dir[0]*math.sin(angle)+self.dir[1]*math.cos(angle)
        )
    def right(self,angle):
        angle *= math.pi/180
        self.dir = (
            self.dir[0]*math.cos(angle)+self.dir[1]*math.sin(angle), 
            -self.dir[0]*math.sin(angle)+self.dir[1]*math.cos(angle)
        )
    def push(self):
        self.stack_pos.append(self.pos)
        self.stack_dir.append(self.dir)
    def pop(self):
        self.pos = self.stack_pos[-1]
        del self.stack_pos[-1]
        self.dir = self.stack_dir[-1]
        del self.stack_dir[-1]
    def point(self):
        return self.pos

class TransformCanvas:
    def __init__(self,xmin,ymin,xmax,ymax,img,scale=0.5):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.rows, self.cols = img.shape[:2]
        self.scale = scale
    def tr(self,p):
        px, py = p
        x = self.scale*(px - self.xmin)
        y = self.scale*(py - self.ymin)
        return (int(x),int(y))

class S:
    def __init__(self,name,params=[]):
        self.name = name
        self.params = params
    def __str__(self):
        s = self.name
        if len(self.params) > 0:
            s += '('
            for i in range(len(self.params)):
                s += ',' if i > 0 else ''
                s += str(self.params[i])
            s += ')'
        return s
    def __repr__(self):
        return str(self)
        
class Lsys: # ROSE LEAF
    def __init__(self,globs=[]):
        self.globs = globs
        w = []
        w.append(S('['))
        w.append(S('{'))
        w.append(S('A',[0,0]))
        w.append(S('.'))
        w.append(S('}'))
        w.append(S(']'))
        w.append(S('['))
        w.append(S('{'))
        w.append(S('A',[0,1]))
        w.append(S('.'))
        w.append(S('}'))
        w.append(S(']'))
        self.word = w
    def step(self):
        w = []
        for s in self.word:
            if s.name == 'A':
                if s.params[1] == 0:
                    w.append(S('.'))
                    w.append(S('G',[self.globs[1],self.globs[2]]))
                    w.append(S('.'))
                    w.append(S('['))
                    w.append(S('+',[self.globs[0]]))
                    w.append(S('B',[s.params[0]]))
                    w.append(S('G',[self.globs[5],self.globs[6],s.params[0]]))
                    w.append(S('.'))
                    w.append(S('}'))
                    w.append(S(']'))
                    w.append(S('['))
                    w.append(S('+',[self.globs[0]]))
                    w.append(S('B',[s.params[0]]))
                    w.append(S('{'))
                    w.append(S('.'))
                    w.append(S(']'))
                    w.append(S('A',[s.params[0]+1,s.params[1]]))
                else:
                    w.append(S('.'))
                    w.append(S('G',[self.globs[1],self.globs[2]]))
                    w.append(S('.'))
                    w.append(S('['))
                    w.append(S('-',[self.globs[0]]))
                    w.append(S('B',[s.params[0]]))
                    w.append(S('G',[self.globs[5],self.globs[6],s.params[0]]))
                    w.append(S('.'))
                    w.append(S('}'))
                    w.append(S(']'))
                    w.append(S('['))
                    w.append(S('-',[self.globs[0]]))
                    w.append(S('B',[s.params[0]]))
                    w.append(S('{'))
                    w.append(S('.'))
                    w.append(S(']'))
                    w.append(S('A',[s.params[0]+1,s.params[1]]))
            elif s.name == 'B':
                if s.params[0] > 0:
                    w.append(S('G',[self.globs[3],self.globs[4]]))
                    w.append(S('B',[s.params[0]-1]))
                else:
                    w.append(s)
            elif s.name == 'G':
                if len(s.params) == 2:
                    w.append(S('G',[s.params[0]*s.params[1],s.params[1]]))
                elif len(s.params) == 3 and s.params[2] > 1:
                    w.append(S('G',[s.params[0]*s.params[1],s.params[1],s.params[2]-1]))
                else:
                    w.append(s)
            else:
                w.append(s)
        self.word = w
    def print(self):
        s = ''
        for w in self.word:
            s += str(w)
        print(s)
    def polygons(self):
        t = Turtle((0,0),(math.cos(math.pi*self.globs[7]/180.0),math.sin(math.pi*self.globs[7]/180.0)))
        for w in self.word:
            if w.name == 'G':
                t.go(w.params[0])
            elif w.name == '+':
                t.left(w.params[0])
            elif w.name == '-':
                t.right(w.params[0])
            elif w.name == '[':
                t.push()
            elif w.name == ']':
                t.pop()
            elif w.name == '.':
                w.point = t.point()            
        polys = []
        for w in self.word:
            if w.name == '{':
                poly = []
            elif w.name == '}':
                polys.append(poly)
            elif w.name == '.':
                poly.append(w.point)
        return polys
    def draw(self,img):
        polys = self.polygons()
        xs = [item[0] for sublist in polys for item in sublist]
        ys = [item[1] for sublist in polys for item in sublist]
        xmin = min(xs)
        xmax = max(xs)
        ymin = min(ys)
        ymax = max(ys)
        canvas = TransformCanvas(xmin,ymin,xmax,ymax,img,)
        for poly in polys:
            poly2 = [ canvas.tr(p) for p in poly ]
            if len(poly2) > 2:
                cv2.fillPoly(img,np.array([poly2]),127)
                cv2.polylines(img,np.array([poly2]),True,255)
            elif len(poly2) == 2:
                cv2.line(img,poly2[0],poly2[1],255,1)
            elif len(poly2) == 1:
                cv2.circle(img,poly2[0],1,255)
            
        oo = canvas.tr((xmax,ymax))
        return (oo[0]<img.shape[1]) and (oo[1]<img.shape[0]) and (oo[0]>=img.shape[1]//6) and (oo[1]>=img.shape[0]//6)

id=0

def gener(params):
    global id
    L = Lsys(params)
    #L.print()
    status = 0
    for n in range(50):
        L.step()
        #L.print()
        img = np.zeros((224,224),np.uint8)
        ok = L.draw(img)
        #print(n,ok)
        #cv2.imshow('leaf',img)
        #cv2.waitKey(0)
        if ok:
            status = 1
            img2 = cv2.resize(img,(28,28))
            img2[img2>0] = 255 # binarize
            cv2.imwrite('dataset/rose{:04d}.png'.format(id),img2)
            np.savetxt('dataset/rose{:04d}.txt'.format(id), np.array(params))
            id += 1
        elif status == 1:
            break
       
if __name__ == "__main__":
    for a in [45,50,55,60,65,70,75,80,85]:
        for g in [4,5,6]:
            for la in [1.15]:
                for lb in [1.3]:
                    for lc in [1.25]:
                        for gg in [3,4]:
                            if gg < g and 2*gg > g:
                                for ld in [1.19]:
                                    for d in [0, 15, 30, 45, 60, 75]:
                                        gener([a,g,la,lb,lc,gg,ld,d])
    #gener([80,5,1.15,1.3,1.25,3,1.19,45])

