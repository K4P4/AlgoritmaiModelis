import math
import random

# GLOBAL VARIABLES:

vertices = []
faces = []

# LITTLE LIBRARY (version 1.1): 

def newface(A,RGB):
  global vertices, faces
  Q = ''
  for i in range (len(A)):
    Q += ' '+str(i+len(vertices))
  faces += [str(len(A))+str(Q)+' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  for i in range (len(A)):
    vertices += [str(A[i][0])+' '+str(A[i][1])+' '+str(A[i][2])]

def cube(c,e,RGB): # c - center, e - edge width, RGB - color
  global vertices, faces
  F = [[0,4,5,1],[0,1,3,2],[0,2,6,4],[1,5,7,3],[2,3,7,6],[4,6,7,5]]
  V = [[0,0,0],[0,0,e],[0,e,0],[0,e,e],[e,0,0],[e,0,e],[e,e,0],[e,e,e]] 
  for i in range (0,6):
    faces += ['4 '+str(F[i][0]+len(vertices))+' '+str(F[i][1]+len(vertices))+' '+str(F[i][2]+len(vertices))+
    ' '+str(F[i][3]+len(vertices))+' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  for j in range (0,8):
    vertices += [str(c[0]+V[j][0]-e/2)+' '+str(c[1]+V[j][1]-e/2)+' '+str(c[2]+V[j][2]-e/2)]

def rectangle3D(c,e,RGB): # c - center, e - width of edges, RGB - color
  global vertices, faces
  F = [[0,4,5,1],[0,1,3,2],[0,2,6,4],[1,5,7,3],[2,3,7,6],[4,6,7,5]]
  V = [[0,0,0],[0,0,e[2]],[0,e[1],0],[0,e[1],e[2]],[e[0],0,0],[e[0],0,e[2]],[e[0],e[1],0],[e[0],e[1],e[2]]] 
  for i in range (0,6):
    faces += ['4 '+str(F[i][0]+len(vertices))+' '+str(F[i][1]+len(vertices))+' '+str(F[i][2]+len(vertices))+
    ' '+str(F[i][3]+len(vertices))+' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  for j in range (0,8):
    vertices += [str(c[0]+V[j][0]-e[0]/2)+' '+str(c[1]+V[j][1]-e[1]/2)+' '+str(c[2]+V[j][2]-e[2]/2)]

def circle(A,B,r,k,RGB): # A - start point, B - end point, r - radius, k - detail, RGB - color
  global vertices, faces
  for i in range (1,k):
    faces += ['3 '+str(2*i+len(vertices))+' '+str(2*k+len(vertices))+
    ' '+str(2*i-2+len(vertices))+' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  faces += ['3 '+str(0+len(vertices))+' '+str(2*k+len(vertices))+
  ' '+str(2*k-2+len(vertices))+' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  if A[0]==B[0] and A[1]==B[1]:
    p1=-math.sqrt(2)*r/2
    p2=(B[2]-A[2])/abs(B[2]-A[2])*p1
    p3=-p1
    p4=p2
    p5=0
  else:
    d=math.sqrt(math.pow(B[0]-A[0],2)+math.pow(B[1]-A[1],2)+math.pow(B[2]-A[2],2))
    f=math.sqrt(math.pow(B[0]-A[0],2)+math.pow(B[1]-A[1],2))
    p1=-r*(B[1]-A[1])/f
    p2=-r*(B[0]-A[0])*(B[2]-A[2])/(d*f)
    p3=r*(B[0]-A[0])/f
    p4=-r*(B[1]-A[1])*(B[2]-A[2])/(d*f)
    p5=r*f/d
  for i in range (1,k+1):
    sinn=math.sin(i/k*2*math.pi)
    coss=math.cos(i/k*2*math.pi)
    q1=coss*p1+sinn*p2
    q2=coss*p3+sinn*p4
    q3=sinn*p5
    vertices += [str(A[0]+q1)+' '+str(A[1]+q2)+' '+str(A[2]+q3)] + [str(B[0]+q1)+' '+str(B[1]+q2)+' '+str(B[2]+q3)]
  vertices += [str(A[0])+' '+str(A[1])+' '+str(A[2])]

def spin3D(A,B,S,min_t,max_t,grid_t,k,RGB): # A - start point, B - end point, r - radius, k - detail, RGB - color, S - parametric function
  global vertices, faces
  def rendervertices(A,B,r):
    global vertices
    if A[0]==B[0] and A[1]==B[1]:
      p1=-math.sqrt(2)*r/2
      p2=(B[2]-A[2])/abs(B[2]-A[2])*p1
      p3=-p1
      p4=p2
      p5=0
    else:
      d=math.sqrt(math.pow(B[0]-A[0],2)+math.pow(B[1]-A[1],2)+math.pow(B[2]-A[2],2))
      f=math.sqrt(math.pow(B[0]-A[0],2)+math.pow(B[1]-A[1],2))
      p1=-r*(B[1]-A[1])/f
      p2=-r*(B[0]-A[0])*(B[2]-A[2])/(d*f)
      p3=r*(B[0]-A[0])/f
      p4=-r*(B[1]-A[1])*(B[2]-A[2])/(d*f)
      p5=r*f/d
    for i in range (1,k+1):
      sinn=math.sin(i/k*2*math.pi)
      coss=math.cos(i/k*2*math.pi)
      q1=coss*p1+sinn*p2
      q2=coss*p3+sinn*p4
      q3=sinn*p5
      vertices += [str(B[0]+q1)+' '+str(B[1]+q2)+' '+str(B[2]+q3)]
  for j in range (grid_t):
    for i in range (k-1):
      faces += ['4 '+str(j*k+i+len(vertices))+' '+str(j*k+i+1+len(vertices))+' '+str((j+1)*k+i+1+len(vertices))+
      ' '+str((j+1)*k+i+len(vertices))+' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
    faces += ['4 '+str((j+1)*k-1+len(vertices))+' '+str(j*k+len(vertices))+' '+str((j+1)*k+len(vertices))+
      ' '+str((j+1)*k-1+k+len(vertices))+' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  AB = math.sqrt(math.pow(B[0]-A[0],2)+math.pow(B[1]-A[1],2)+math.pow(B[2]-A[2],2))
  for j in range (grid_t+1):
    t = min_t+(max_t-min_t)/grid_t*j
    g = S(t)
    C = [None,None,None]
    for h in range (3):
      C[h] = A[h] + (B[h]-A[h]) * g[1] / AB
    if A[0] == C[0] and A[1] == C[1] and A[2] == C[2]:
      rendervertices([2*C[0]-B[0],2*C[1]-B[1],2*C[2]-B[2]],C,g[0])  
    else:
      if g[1]<=0:
        rendervertices([2*C[0]-A[0],2*C[1]-A[1],2*C[2]-A[2]],C,g[0])
      else:
        rendervertices(A,C,g[0])
	
def pyramid(c,e,h,RGB): # c - center, e - edge width, h - high, RGB - color
  global vertices, faces
  F = [[[0,1,2,3],[1,4,2],[3,2,4],[0,3,4],[0,4,1]],[[3,2,1,0],[2,4,1],[4,2,3],[4,3,0],[1,4,0]]]
  V = [[0,0,0],[e,0,0],[e,0,e],[0,0,e],[e/2,h,e/2]]
  idx = 0
  if h < 0:
    idx = 1
  for i in range (1,5):
    faces += ['3 '+str(F[idx][i][0]+len(vertices))+' '+str(F[idx][i][1]+len(vertices))+' '+str(F[idx][i][2]+len(vertices))+
    ' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  faces += ['4 '+str(F[idx][0][0]+len(vertices))+' '+str(F[idx][0][1]+len(vertices))+' '+str(F[idx][0][2]+len(vertices))+
    ' '+str(F[idx][0][3]+len(vertices))+' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  for j in range (0,5):
    vertices += [str(c[0]+V[j][0]-e/2)+' '+str(c[1]+V[j][1]-e/2)+' '+str(c[2]+V[j][2]-e/2)]

def cube2(c,e,b,RGB): # c - center, b - border width, e - edge width, RGB - color
  global vertices, faces
  F = [[2,3,9,7],[1,6,9,3],[7,5,0,2],[0,5,6,1],[8,9,12,13],[12,9,6,10],[8,13,11,5],[6,5,11,10],
  [19,14,12,18],[12,10,15,18],[11,14,19,16],[10,11,16,15],[4,17,18,3],[3,18,15,1],[1,15,16,0],[0,16,17,4],
  [21,22,4,3],[21,3,2,20],[22,23,0,4],[23,20,2,0],[26,27,9,8],[7,9,27,25],[8,5,28,26],[25,28,5,7],
  [32,30,12,14],[12,30,31,13],[31,34,11,13],[34,32,14,11],[37,19,18,36],[35,36,18,17],[39,35,17,16],[39,16,19,37],
  [21,24,38,36],[22,21,36,35],[23,22,35,39],[24,23,39,38],[27,29,24,21],[21,20,25,27],[23,28,25,20],[23,24,29,28],
  [29,27,30,33],[27,26,31,30],[28,29,33,34],[26,28,34,31],[36,38,33,30],[37,36,30,32],[34,33,38,39],[39,37,32,34]]
  V = [[0,0,0],[0,b,b],[b,0,b],[b,b,b],[b,b,0],[0,0,e],[0,b,e-b],[b,0,e-b],
  [b,b,e],[b,b,e-b],[0,e-b,e-b],[0,e,e],[b,e-b,e-b],[b,e-b,e],[b,e,e-b],[0,e-b,b],
  [0,e,0],[b,e-b,0],[b,e-b,b],[b,e,b],[e-b,0,b],[e-b,b,b],[e-b,b,0],[e,0,0],
  [e,b,b],[e-b,0,e-b],[e-b,b,e],[e-b,b,e-b],[e,0,e],[e,b,e-b],[e-b,e-b,e-b],[e-b,e-b,e],
  [e-b,e,e-b],[e,e-b,e-b],[e,e,e],[e-b,e-b,0],[e-b,e-b,b],[e-b,e,b],[e,e-b,b],[e,e,0]] 
  for i in range (0,48):
    faces += ['4 '+str(F[i][0]+len(vertices))+' '+str(F[i][1]+len(vertices))+' '+str(F[i][2]+len(vertices))+
    ' '+str(F[i][3]+len(vertices))+' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  for j in range (0,40):
    vertices += [str(c[0]+V[j][0]-e/2)+' '+str(c[1]+V[j][1]-e/2)+' '+str(c[2]+V[j][2]-e/2)]

def parametric(S,min_u,max_u,grid_u,min_v,max_v,grid_v,RGB): # S - parametric uv surface, grid - detail, RGB - color
  global vertices, faces
  for i in range (grid_u):
    for j in range (grid_v):
      A = i*(grid_v+1)+j
      B = A+grid_v+1
      faces += ['4 '+str(A+len(vertices))+' '+str(B+len(vertices))+' '+str(B+1+len(vertices))+
      ' '+str(A+1+len(vertices))+' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  for i in range (grid_u+1):
    for j in range (grid_v+1):
      u = min_u+(max_u-min_u)/grid_u*i
      v = min_v+(max_v-min_v)/grid_v*j
      f = S(u,v)
      vertices += [str(f[0])+' '+str(f[1])+' '+str(f[2])]

def sphere(c,r,k,RGB): # c - center, r - radius, k - detail, RGB - color
  global vertices, faces
  M = [[None for j in range(k+1)] for i in range(k)]
  def render1(pr,k,st,RGB):
    global faces
    Q = [[None for j in range(k+1)] for i in range(k+1)]
    for i in range (k):
      for j in range (k+1):
        Q[i][j] = pr+j+i*(k+1)
    for i in range (k+1):
      Q[k][i] = st+i+1
    for i in range (k):
      for j in range (k):
        faces += ['4 '+str(Q[i][j]+len(vertices))+' '+str(Q[i+1][j]+len(vertices))+' '+str(Q[i+1][j+1]+len(vertices))+
        ' '+str(Q[i][j+1]+len(vertices))+' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  def render2(pr,k,RGB):
    global faces
    Q = [[None for j in range(k-1)] for i in range(k-1)]
    for i in range (k-1):
      for j in range (k-1):
        Q[i][j] = pr+j+i*(k-1)
    for i in range (k-2):
      for j in range (k-2):
        faces += ['4 '+str(Q[i][j]+len(vertices))+' '+str(Q[i+1][j]+len(vertices))+' '+str(Q[i+1][j+1]+len(vertices))+
        ' '+str(Q[i][j+1]+len(vertices))+' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  def render3(a,b,c,d,RGB):
    global faces
    faces += ['4 '+str(a+len(vertices))+' '+str(b+len(vertices))+' '+str(c+len(vertices))+' '+str(d+len(vertices))+
    ' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  def render4(a1,a2,b1,b2,k,RGB):
    global faces
    p = a2-a1
    q = b2-b1
    for i in range (k-2):
      faces += ['4 '+str(a1+i*p+len(vertices))+' '+str(a1+(i+1)*p+len(vertices))+' '+str(b1+(i+1)*q+len(vertices))+
      ' '+str(b1+i*q+len(vertices))+' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  render1(0,k,math.pow(k,2)+k-1,RGB)
  render1(k*(k+1),k,2*math.pow(k,2)+2*k-1,RGB)
  render1(2*k*(k+1),k,3*math.pow(k,2)+3*k-1,RGB)
  render1(3*k*(k+1),k,-1,RGB)
  if k == 1:
    faces += ['4 '+str(0+len(vertices))+' '+str(6+len(vertices))+' '+str(4+len(vertices))+' '+str(2+len(vertices))+
    ' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
    faces += ['4 '+str(1+len(vertices))+' '+str(3+len(vertices))+' '+str(5+len(vertices))+' '+str(7+len(vertices))+
    ' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  else:
    render2(4*k*(k+1),k,RGB)
    render2(5*math.pow(k,2)+2*k+1,k,RGB)
    render3(k+1,0,(k+1)*(4*k-1),4*math.pow(k,2)+5*k-2,RGB)
    render3(math.pow(k+1,2),k*(k+1),(k-1)*(k+1),k*(5*k+2),RGB)
    render3((k+1)*(2*k+1),2*k*(k+1),(k+1)*(2*k-1),5*math.pow(k,2)+k+2,RGB)
    render3((k+1)*(3*k+1),3*k*(k+1),(k+1)*(3*k-1),4*k*(k+1),RGB)
    render3(4*math.pow(k,2)+4*k-1,k,2*k+1,5*math.pow(k,2)+2*k+1,RGB)
    render3(3*math.pow(k,2)+3*k-1,k*(3*k+4),3*math.pow(k,2)+5*k+1,5*math.pow(k,2)+3*k-1,RGB)
    render3(2*math.pow(k,2)+2*k-1,k*(2*k+3),2*math.pow(k,2)+4*k+1,6*math.pow(k,2)+1,RGB)
    render3(math.pow(k,2)+k-1,k*(k+2),math.pow(k,2)+3*k+1,6*math.pow(k,2)-k+3,RGB)
    render4(4*math.pow(k,2)+5*k-2,4*math.pow(k,2)+6*k-3,k+1,2*k+2,k,RGB)
    render4(k*(5*k+2),5*math.pow(k,2)+2*k-1,math.pow(k+1,2),(k+2)*(k+1),k,RGB)
    render4(5*math.pow(k,2)+k+2,5*math.pow(k,2)+3,(k+1)*(2*k+1),2*math.pow(k+1,2),k,RGB)
    render4(4*k*(k+1),math.pow(2*k+1,2),(k+1)*(3*k+1),(k+1)*(3*k+2),k,RGB)
    render4(2*k+1,3*k+2,5*math.pow(k,2)+2*k+1,k*(5*k+3),k,RGB)
    render4(math.pow(k,2)+3*k+1,math.pow(k,2)+4*k+2,6*math.pow(k,2)-k+3,6*math.pow(k,2)-k+4,k,RGB)
    render4(2*math.pow(k,2)+4*k+1,(k+2)*(2*k+1),6*math.pow(k,2)+1,6*math.pow(k,2)-k+2,k,RGB)
    render4(3*math.pow(k,2)+5*k+1,3*math.pow(k,2)+6*k+2,5*math.pow(k,2)+3*k-1,(k+1)*(5*k-2),k,RGB)
  for i in range (k):
    for j in range (k+1):
      x = 1/math.tan(math.pi/4+math.pi/2*i/k)
      y = 1/math.tan(math.pi/4+math.pi/2*j/k)
      d = r/math.sqrt(math.pow(x,2)+math.pow(y,2)+1)
      M[i][j] = [x*d,y*d,d]
      vertices += [str(c[0]+M[i][j][0])+' '+str(c[1]+M[i][j][1])+' '+str(c[2]+M[i][j][2])]
  for i in range (k):
    for j in range (k+1):
      vertices += [str(c[0]-M[i][j][2])+' '+str(c[1]+M[i][j][1])+' '+str(c[2]+M[i][j][0])]
  for i in range (k):
    for j in range (k+1):
      vertices += [str(c[0]-M[i][j][0])+' '+str(c[1]+M[i][j][1])+' '+str(c[2]-M[i][j][2])]
  for i in range (k):
    for j in range (k+1):
      vertices += [str(c[0]+M[i][j][2])+' '+str(c[1]+M[i][j][1])+' '+str(c[2]-M[i][j][0])]
  for i in range (1,k):
    for j in range (1,k):
      vertices += [str(c[0]+M[i][j][0])+' '+str(c[1]+M[i][j][2])+' '+str(c[2]-M[i][j][1])]
  for i in range (1,k):
    for j in range (1,k):
      vertices += [str(c[0]+M[i][j][0])+' '+str(c[1]-M[i][j][2])+' '+str(c[2]+M[i][j][1])]

def cylinder(A,B,r,k,RGB): # A - start point, B - end point, r - radius, k - detail, RGB - color
  global vertices, faces
  for i in range (1,k):
    faces += ['4 '+str(2*i-2+len(vertices))+' '+str(2*i+len(vertices))+' '+str(2*i+1+len(vertices))+
    ' '+str(2*i-1+len(vertices))+' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
    faces += ['3 '+str(2*i+len(vertices))+' '+str(2*i-2+len(vertices))+' '+str(2*k+len(vertices))+
    ' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
    faces += ['3 '+str(2*i-1+len(vertices))+' '+str(2*i+1+len(vertices))+' '+str(2*k+1+len(vertices))+
    ' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  faces += ['4 '+str(2*k-2+len(vertices))+' '+str(0+len(vertices))+' '+str(1+len(vertices))+
  ' '+str(2*k-1+len(vertices))+' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  faces += ['3 '+str(0+len(vertices))+' '+str(2*k-2+len(vertices))+' '+str(2*k+len(vertices))+
  ' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  faces += ['3 '+str(1+len(vertices))+' '+str(2*k+1+len(vertices))+' '+str(2*k-1+len(vertices))+
  ' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  if A[0]==B[0] and A[1]==B[1]:
    p1=-math.sqrt(2)*r/2
    p2=(B[2]-A[2])/abs(B[2]-A[2])*p1
    p3=-p1
    p4=p2
    p5=0
  else:
    d=math.sqrt(math.pow(B[0]-A[0],2)+math.pow(B[1]-A[1],2)+math.pow(B[2]-A[2],2))
    f=math.sqrt(math.pow(B[0]-A[0],2)+math.pow(B[1]-A[1],2))
    p1=-r*(B[1]-A[1])/f
    p2=-r*(B[0]-A[0])*(B[2]-A[2])/(d*f)
    p3=r*(B[0]-A[0])/f
    p4=-r*(B[1]-A[1])*(B[2]-A[2])/(d*f)
    p5=r*f/d
  for i in range (1,k+1):
    sinn=math.sin(i/k*2*math.pi)
    coss=math.cos(i/k*2*math.pi)
    q1=coss*p1+sinn*p2
    q2=coss*p3+sinn*p4
    q3=sinn*p5
    vertices += [str(A[0]+q1)+' '+str(A[1]+q2)+' '+str(A[2]+q3)] + [str(B[0]+q1)+' '+str(B[1]+q2)+' '+str(B[2]+q3)]
  vertices += [str(A[0])+' '+str(A[1])+' '+str(A[2])] + [str(B[0])+' '+str(B[1])+' '+str(B[2])]

def cylinder2(A,B,r,k,RGB): # A - start point, B - end point, r - radius, k - detail, RGB - color
  global vertices, faces
  for i in range (1,k):
    faces += ['4 '+str(2*i-2+len(vertices))+' '+str(2*i+len(vertices))+' '+str(2*i+1+len(vertices))+
    ' '+str(2*i-1+len(vertices))+' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  faces += ['4 '+str(2*k-2+len(vertices))+' '+str(0+len(vertices))+' '+str(1+len(vertices))+
  ' '+str(2*k-1+len(vertices))+' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  if A[0]==B[0] and A[1]==B[1]:
    p1=-math.sqrt(2)*r/2
    p2=(B[2]-A[2])/abs(B[2]-A[2])*p1
    p3=-p1
    p4=p2
    p5=0
  else:
    d=math.sqrt(math.pow(B[0]-A[0],2)+math.pow(B[1]-A[1],2)+math.pow(B[2]-A[2],2))
    f=math.sqrt(math.pow(B[0]-A[0],2)+math.pow(B[1]-A[1],2))
    p1=-r*(B[1]-A[1])/f
    p2=-r*(B[0]-A[0])*(B[2]-A[2])/(d*f)
    p3=r*(B[0]-A[0])/f
    p4=-r*(B[1]-A[1])*(B[2]-A[2])/(d*f)
    p5=r*f/d
  for i in range (1,k+1):
    sinn=math.sin(i/k*2*math.pi)
    coss=math.cos(i/k*2*math.pi)
    q1=coss*p1+sinn*p2
    q2=coss*p3+sinn*p4
    q3=sinn*p5
    vertices += [str(A[0]+q1)+' '+str(A[1]+q2)+' '+str(A[2]+q3)] + [str(B[0]+q1)+' '+str(B[1]+q2)+' '+str(B[2]+q3)]

def cylinder3(A,B,r,k,RGB): # A - start point, B - end point, r - radius, k - detail, RGB - color
  global vertices, faces
  for i in range (1,k):
    faces += ['4 '+str(2*i-2+len(vertices))+' '+str(2*i+len(vertices))+' '+str(2*i+1+len(vertices))+
    ' '+str(2*i-1+len(vertices))+' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
    faces += ['3 '+str(2*i+len(vertices))+' '+str(2*i-2+len(vertices))+' '+str(2*k+len(vertices))+
    ' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  faces += ['4 '+str(2*k-2+len(vertices))+' '+str(0+len(vertices))+' '+str(1+len(vertices))+
  ' '+str(2*k-1+len(vertices))+' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  faces += ['3 '+str(0+len(vertices))+' '+str(2*k-2+len(vertices))+' '+str(2*k+len(vertices))+
  ' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  if A[0]==B[0] and A[1]==B[1]:
    p1=-math.sqrt(2)*r/2
    p2=(B[2]-A[2])/abs(B[2]-A[2])*p1
    p3=-p1
    p4=p2
    p5=0
  else:
    d=math.sqrt(math.pow(B[0]-A[0],2)+math.pow(B[1]-A[1],2)+math.pow(B[2]-A[2],2))
    f=math.sqrt(math.pow(B[0]-A[0],2)+math.pow(B[1]-A[1],2))
    p1=-r*(B[1]-A[1])/f
    p2=-r*(B[0]-A[0])*(B[2]-A[2])/(d*f)
    p3=r*(B[0]-A[0])/f
    p4=-r*(B[1]-A[1])*(B[2]-A[2])/(d*f)
    p5=r*f/d
  for i in range (1,k+1):
    sinn=math.sin(i/k*2*math.pi)
    coss=math.cos(i/k*2*math.pi)
    q1=coss*p1+sinn*p2
    q2=coss*p3+sinn*p4
    q3=sinn*p5
    vertices += [str(A[0]+q1)+' '+str(A[1]+q2)+' '+str(A[2]+q3)] + [str(B[0]+q1)+' '+str(B[1]+q2)+' '+str(B[2]+q3)]
  vertices += [str(A[0])+' '+str(A[1])+' '+str(A[2])]

def cone(A,B,r,k,RGB): # A - start point, B - end point, r - radius, k - detail, RGB - color
  global vertices, faces
  for i in range (1,k):
    faces += ['3 '+str(i+len(vertices))+' '+str(i-1+len(vertices))+' '+str(k+len(vertices))+
    ' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
    faces += ['3 '+str(i-1+len(vertices))+' '+str(i+len(vertices))+' '+str(k+1+len(vertices))+
    ' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  faces += ['3 '+str(0+len(vertices))+' '+str(k-1+len(vertices))+' '+str(k+len(vertices))+
  ' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  faces += ['3 '+str(k-1+len(vertices))+' '+str(0+len(vertices))+' '+str(k+1+len(vertices))+
  ' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  if A[0]==B[0] and A[1]==B[1]:
    p1=-math.sqrt(2)*r/2
    p2=(B[2]-A[2])/abs(B[2]-A[2])*p1
    p3=-p1
    p4=p2
    p5=0
  else:
    d=math.sqrt(math.pow(B[0]-A[0],2)+math.pow(B[1]-A[1],2)+math.pow(B[2]-A[2],2))
    f=math.sqrt(math.pow(B[0]-A[0],2)+math.pow(B[1]-A[1],2))
    p1=-r*(B[1]-A[1])/f
    p2=-r*(B[0]-A[0])*(B[2]-A[2])/(d*f)
    p3=r*(B[0]-A[0])/f
    p4=-r*(B[1]-A[1])*(B[2]-A[2])/(d*f)
    p5=r*f/d
  for i in range (1,k+1):
    sinn=math.sin(i/k*2*math.pi)
    coss=math.cos(i/k*2*math.pi)
    q1=coss*p1+sinn*p2
    q2=coss*p3+sinn*p4
    q3=sinn*p5
    vertices += [str(A[0]+q1)+' '+str(A[1]+q2)+' '+str(A[2]+q3)]
  vertices += [str(A[0])+' '+str(A[1])+' '+str(A[2])] + [str(B[0])+' '+str(B[1])+' '+str(B[2])]

def cone2(A,B,r,k,RGB): # A - start point, B - end point, r - radius, k - detail, RGB - color
  global vertices, faces
  for i in range (1,k):
    faces += ['3 '+str(i-1+len(vertices))+' '+str(i+len(vertices))+' '+str(k+len(vertices))+
    ' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  faces += ['3 '+str(k-1+len(vertices))+' '+str(0+len(vertices))+' '+str(k+len(vertices))+
  ' '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])]
  if A[0]==B[0] and A[1]==B[1]:
    p1=-math.sqrt(2)*r/2
    p2=(B[2]-A[2])/abs(B[2]-A[2])*p1
    p3=-p1
    p4=p2
    p5=0
  else:
    d=math.sqrt(math.pow(B[0]-A[0],2)+math.pow(B[1]-A[1],2)+math.pow(B[2]-A[2],2))
    f=math.sqrt(math.pow(B[0]-A[0],2)+math.pow(B[1]-A[1],2))
    p1=-r*(B[1]-A[1])/f
    p2=-r*(B[0]-A[0])*(B[2]-A[2])/(d*f)
    p3=r*(B[0]-A[0])/f
    p4=-r*(B[1]-A[1])*(B[2]-A[2])/(d*f)
    p5=r*f/d
  for i in range (1,k+1):
    sinn=math.sin(i/k*2*math.pi)
    coss=math.cos(i/k*2*math.pi)
    q1=coss*p1+sinn*p2
    q2=coss*p3+sinn*p4
    q3=sinn*p5
    vertices += [str(A[0]+q1)+' '+str(A[1]+q2)+' '+str(A[2]+q3)]
  vertices += [str(B[0])+' '+str(B[1])+' '+str(B[2])]

def clear():
  global vertices, faces
  vertices = []
  faces = []

def off(mesh): # mesh - off file
  global vertices, faces
  file = open(mesh, 'w')
  file.write('%s\n%d %d %d\n' % ('OFF',len(vertices),len(faces),0))
  for i in range (len(vertices)):
    file.write('%s\n' % vertices[i])
  for j in range (len(faces)):
    file.write('%s\n' % faces[j])
  file.close()
  clear()

# EXAMPLES:

def example1():
  m = 10
  for k in range (m):
    for i in range (m-k):
      for j in range (m-k):
        if i == 0 or i == m-k-1 or j == 0 or j == m-k-1:
          cube([i+k/2,k/2,j+k/2],0.8,[random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)])
          if k != 0:
            cube([i+k/2,-k/2,j+k/2],0.8,[random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)])
  off('example1.off')

def example2():
  m = 10
  for k in range (m):
    for i in range (m-k):
      for j in range (m-k):
        if i == 0 or i == m-k-1 or j == 0 or j == m-k-1:
          cube2([i+k/2,k/2,j+k/2],0.8,0.1,[random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)])
          if k != 0:
            cube2([i+k/2,-k/2,j+k/2],0.8,0.1,[random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)])
  off('example2.off')

def example3():
  a = 5
  b = 1
  def toras1(u,v):
    x = (a+b*math.cos(u))*math.cos(v)
    y = b*math.sin(u)
    z = (a+b*math.cos(u))*math.sin(v)
    return ([x, y, z])
  def toras2(u,v):
    x = (a+b*math.cos(u))*math.cos(v)
    y = -(a+b*math.cos(u))*math.sin(v)
    z = b*math.sin(u)
    return ([x, y, z])
  def toras3(u,v):
    x = b*math.sin(u)
    y = -(a+b*math.cos(u))*math.cos(v)
    z = (a+b*math.cos(u))*math.sin(v)
    return ([x, y, z])
  parametric(toras1,0,2*math.pi,50,0,2*math.pi,200,[0,255,0])
  parametric(toras2,0,2*math.pi,50,0,2*math.pi,200,[255,0,0])
  parametric(toras3,0,2*math.pi,50,0,2*math.pi,200,[0,0,255])
  off('example3.off')

def example4():
  a = 36
  b = 0.85
  c = 1
  h = 15
  r = 0.15
  s = 0.5
  def sakos(u,v):
    x = math.sqrt(u)*math.cos(u)*v
    y = h-h/a*u
    z = math.sqrt(u)*math.sin(u)*v
    return ([x, y, z])
  def virsune(u,v):
    w = (3/2)*math.sqrt(3)
    x = c*math.sqrt(1-u*u)*(1-u)/w*math.cos(v)
    y = h+c*u+c
    z = c*math.sqrt(1-u*u)*(1-u)/w*math.sin(v)
    return ([x, y, z])
  def vamzdis(u,v):
    x = math.sqrt(b)*math.cos(a*b)*(-s*u+u*math.sqrt(a)+s)+r*math.cos(v)*math.sin(a*b)
    y = -r*math.sin(v)-b*h+h
    z = math.sqrt(b)*math.sin(a*b)*(-s*u+u*math.sqrt(a)+s)-r*math.cos(v)*math.cos(a*b)
    return ([x, y, z])
  def papuosimai(u,v):
    x = math.sqrt(u)*math.cos(u)-r*(math.cos(v)*(2*u*math.cos(u)+math.sin(u))/math.sqrt(4*u*u+1)-2*h*r*math.sqrt(u)*math.sin(v)*(2*u*math.sin(u)-math.cos(u))/math.sqrt((4*a*a*u*u+4*h*h*u+a*a)*(4*u*u+1)))
    y = h*(a-u)/a+a*r*math.sin(v)*math.sqrt(4*u*u+1)/math.sqrt(4*a*a*u*u+4*h*h*u+a*a)
    z = math.sqrt(u)*math.sin(u)-r*(math.cos(v)*(2*u*math.sin(u)-math.cos(u))/math.sqrt(4*u*u+1)-2*r*math.sqrt(u)*math.sin(v)*(2*u*math.cos(u)+math.sin(u))*h/math.sqrt((4*a*a*u*u+4*h*h*u+a*a)*(4*u*u+1)))
    return ([x, y, z])
  def stiebas(u,v):
    x = s*math.cos(u)*math.sqrt(v/h)
    y = h-v
    z = s*math.sin(u)*math.sqrt(v/h)
    return ([x, y, z])
  def dugnas(u,v):
    x = u*math.cos(v)
    y = 0
    z = u*math.sin(v)
    return ([x, y, z])
  def sfera1(u,v):
    x = r*math.cos(u)*math.sin(v)
    y = r*math.cos(v)+h
    z = r*math.sin(u)*math.sin(v)
    return ([x, y, z])
  def sfera2(u,v):
    x = math.sqrt(a*b)*math.cos(a*b)-r*math.cos(u)*math.sin(v)
    y = h*(1-b)+r*math.cos(v)
    z = math.sqrt(a*b)*math.sin(a*b)-r*math.sin(u)*math.sin(v)
    return ([x, y, z])
  parametric(sakos,0,a*b,500,0.98*s/math.sqrt(a),1,15,[0,255,0])
  parametric(virsune,-1,1,50,0,2*math.pi,40,[255,0,0])
  parametric(vamzdis,-0.01,1,5,0,2*math.pi,20,[255,255,255])
  parametric(papuosimai,0,a*b,500,0,2*math.pi,20,[255,255,255])
  parametric(stiebas,0,2*math.pi,20,0,h,100,[139,69,19])
  parametric(dugnas,0,s,1,0,2*math.pi,20,[139,69,19])
  parametric(sfera1,0,2*math.pi,30,0,math.pi,30,[255,255,255])
  parametric(sfera2,0,2*math.pi,30,0,math.pi,30,[255,255,255])
  off('example4.off')

def example5():
  m = 7
  for i in range (m):
    for j in range (m-i):
      for k in range (m-i-j):
        sphere([i*math.sqrt(3)/2+(k-1)*math.sqrt(3)/6,k*math.sqrt(2/3),j+0.5*(i-1)+(k-1)/2],0.5,10,[random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)])
  off('example5.off')

def example6():
  V = [[-0.262865,0,0.425325],[0.262865,0,0.425325],[-0.262865,0,-0.425325],[0.262865,0,-0.425325],
  [0,0.425325,0.262865],[0,0.425325,-0.262865],[0,-0.425325,0.262865],[0,-0.425325,-0.262865],
  [0.425325,0.262865,0],[-0.425325,0.262865,0],[0.425325,-0.262865,0],[-0.425325,-0.262865,0]]
  E = [[0,1],[0,4],[0,6],[0,9],[0,11],[1,4],[1,6],[1,8],[1,10],[2,3],[2,5],[2,7],[2,9],[2,11],[3,5],
  [3,7],[3,8],[3,10],[4,5],[4,8],[4,9],[5,8],[5,9],[6,7],[6,10],[6,11],[7,10],[7,11],[8,10],[9,11]]
  for v in V:
    sphere(v,0.06,10,[0,255,0])
  for e in E:
    cylinder2(V[e[0]],V[e[1]],0.02,15,[0,0,255])
  off('example6.off')

def example7():
  V = [[-1.411334,3.199887,0],[-.705666,3.199887,1.222252],[.705666,3.199887,1.222252],[1.411334,3.199887,0],
  [.705666,3.199887,-1.22225],[-.705666,3.199887,-1.22225],[-2.55312,2.385067,-.155618],[-2.98926,1.570227,.911012],
  [-2.28358,1.570227,2.133254],[-1.141792,2.385067,2.288874],[-1.411334,2.385067,-2.133266],[-2.55312,1.881465,-1.474048],
  [-.705666,1.570228,-3.044266],[1.411334,2.385067,-2.133266],[.705666,1.570228,-3.044266],[-1.141792,.251797,-3.296066],
  [-2.28358,-.251797,-2.636866],[-2.98926,.563037,-1.725844],[0,-.563034,-3.451686],[0,-1.881465,-2.948087],
  [-1.141792,-2.385053,-2.288887],[-2.28358,-1.570225,-2.133266],[1.141792,.251797,-3.296066],[2.28358,-.251797,-2.636866],
  [2.28358,-1.570225,-2.133266],[1.141792,-2.385053,-2.288887],[2.55312,1.881465,-1.474048],[2.98926,.563037,-1.725844],
  [2.55312,2.385067,-.155618],[2.98926,1.570227,.911012],[3.42538,.251797,.659216],[3.42538,-.251797,-.659214],
  [1.141792,2.385067,2.288874],[2.28358,1.570227,2.133254],[0,1.881464,2.948094],[-2.28358,.251797,2.636854],
  [-1.141792,-.251798,3.296074],[0,.563036,3.451694],[-3.42538,.251797,.659216],[-2.98926,-.563035,1.725846],
  [-3.42538,-.251797,-.659214],[-2.98926,-1.570225,-.91101],[-2.55312,-2.385053,.15562],[-2.55312,-1.881465,1.474048],
  [-.705666,-3.199893,-1.222251],[-1.411334,-3.199893,0],[.705666,-3.199893,-1.222251],[2.98926,-1.570225,-.91101],
  [2.55312,-2.385053,.15562],[1.411334,-3.199893,0],[2.98926,-.563035,1.725846],[2.55312,-1.881465,1.474048],
  [2.28358,.251797,2.636854],[1.141792,-.251798,3.296074],[.705666,-1.570226,3.044274],[1.411334,-2.385053,2.133253],
  [-.705666,-1.570226,3.044274],[-1.411334,-2.385053,2.133253],[-.705666,-3.199893,1.222251],[.705666,-3.199893,1.222251]]
  E = [[0,1],[0,5],[0,6],[1,2],[1,9],[2,3],[2,32],[3,4],[3,28],[4,5],[4,13],[5,10],[6,7],[6,11],[7,8],[7,38],[8,9],
  [8,35],[9,34],[10,11],[10,12],[11,17],[12,14],[12,15],[13,14],[13,26],[14,22],[15,16],[15,18],[16,17],[16,21],[17,40],
  [18,19],[18,22],[19,20],[19,25],[20,21],[20,44],[21,41],[22,23],[23,24],[23,27],[24,25],[24,47],[25,46],[26,27],[26,28],
  [27,31],[28,29],[29,30],[29,33],[30,31],[30,50],[31,47],[32,33],[32,34],[33,52],[34,37],[35,36],[35,39],[36,37],[36,56],
  [37,53],[38,39],[38,40],[39,43],[40,41],[41,42],[42,43],[42,45],[43,57],[44,45],[44,46],[45,58],[46,49],[47,48],[48,49],
  [48,51],[49,59],[50,51],[50,52],[51,55],[52,53],[53,54],[54,55],[54,56],[55,59],[56,57],[57,58],[58,59]]
  for v in V:
    sphere(v,0.2,10,[0,255,0])
  for e in E:
    cylinder2(V[e[0]],V[e[1]],0.08,15,[0,0,255])
  off('example7.off')

def example8():
  V = [[-1.411334,3.199887,0],[-.705666,3.199887,1.222252],[.705666,3.199887,1.222252],[1.411334,3.199887,0],
  [.705666,3.199887,-1.22225],[-.705666,3.199887,-1.22225],[-2.55312,2.385067,-.155618],[-2.98926,1.570227,.911012],
  [-2.28358,1.570227,2.133254],[-1.141792,2.385067,2.288874],[-1.411334,2.385067,-2.133266],[-2.55312,1.881465,-1.474048],
  [-.705666,1.570228,-3.044266],[1.411334,2.385067,-2.133266],[.705666,1.570228,-3.044266],[-1.141792,.251797,-3.296066],
  [-2.28358,-.251797,-2.636866],[-2.98926,.563037,-1.725844],[0,-.563034,-3.451686],[0,-1.881465,-2.948087],
  [-1.141792,-2.385053,-2.288887],[-2.28358,-1.570225,-2.133266],[1.141792,.251797,-3.296066],[2.28358,-.251797,-2.636866],
  [2.28358,-1.570225,-2.133266],[1.141792,-2.385053,-2.288887],[2.55312,1.881465,-1.474048],[2.98926,.563037,-1.725844],
  [2.55312,2.385067,-.155618],[2.98926,1.570227,.911012],[3.42538,.251797,.659216],[3.42538,-.251797,-.659214],
  [1.141792,2.385067,2.288874],[2.28358,1.570227,2.133254],[0,1.881464,2.948094],[-2.28358,.251797,2.636854],
  [-1.141792,-.251798,3.296074],[0,.563036,3.451694],[-3.42538,.251797,.659216],[-2.98926,-.563035,1.725846],
  [-3.42538,-.251797,-.659214],[-2.98926,-1.570225,-.91101],[-2.55312,-2.385053,.15562],[-2.55312,-1.881465,1.474048],
  [-.705666,-3.199893,-1.222251],[-1.411334,-3.199893,0],[.705666,-3.199893,-1.222251],[2.98926,-1.570225,-.91101],
  [2.55312,-2.385053,.15562],[1.411334,-3.199893,0],[2.98926,-.563035,1.725846],[2.55312,-1.881465,1.474048],
  [2.28358,.251797,2.636854],[1.141792,-.251798,3.296074],[.705666,-1.570226,3.044274],[1.411334,-2.385053,2.133253],
  [-.705666,-1.570226,3.044274],[-1.411334,-2.385053,2.133253],[-.705666,-3.199893,1.222251],[.705666,-3.199893,1.222251]]
  sphere([0,0,0],1.5,15,[0,0,255])
  for v in V:
    cylinder2([0,0,0],v,0.2,15,[255,255,0])
    cone(v,[1.3*v[0],1.3*v[1],1.3*v[2]],0.5,15,[255,0,0])
  off('example8.off')