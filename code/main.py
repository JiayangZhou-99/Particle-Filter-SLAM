"""
Created on Sun Feb 20 17:40:04 2022

@author: zhoujiayang
"""


import pandas as pd
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
import time
import collections
import random

def tic():
  return time.time()
def toc(tstart, name="Operation"):
  print('%s took: %s sec.\n' % (name,(time.time() - tstart)))


def compute_stereo():
  path_l = 'data/image_left.png'
  path_r = 'data/image_right.png'

  image_l = cv2.imread(path_l, 0)
  image_r = cv2.imread(path_r, 0)

  image_l = cv2.cvtColor(image_l, cv2.COLOR_BAYER_BG2BGR)
  image_r = cv2.cvtColor(image_r, cv2.COLOR_BAYER_BG2BGR)

  image_l_gray = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY)
  image_r_gray = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)

  # You may need to fine-tune the variables `numDisparities` and `blockSize` based on the desired accuracy
  stereo = cv2.StereoBM_create(numDisparities=32, blockSize=9) 
  disparity = stereo.compute(image_l_gray, image_r_gray)

  fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
  ax1.imshow(image_l)
  ax1.set_title('Left Image')
  ax2.imshow(image_r)
  ax2.set_title('Right Image')
  ax3.imshow(disparity, cmap='gray')
  ax3.set_title('Disparity Map')
  plt.show()
  

def read_data_from_csv(filename):
  '''
  INPUT 
  filename        file address

  OUTPUT 
  timestamp       timestamp of each observation
  data            a numpy array containing a sensor measurement in each row
  '''
  data_csv = pd.read_csv(filename, header=None)
  data = data_csv.values[:, 1:]
  timestamp = data_csv.values[:, 0]
  return timestamp, data


def mapCorrelation(im, x_im, y_im, vp, xs, ys):
  '''
  INPUT 
  im              the map 
  x_im,y_im       physical x,y positions of the grid map cells
  vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)  
  xs,ys           physical x,y,positions you want to evaluate "correlation" 

  OUTPUT 
  c               sum of the cell values of all the positions hit by range sensor

  '''
  MAP= {}
  MAP['res']   = 1 #meters
  MAP['xmin']  = 0  #meters
  MAP['ymin']  = -1000-100
  MAP['xmax']  = 1200 + 100
  MAP['ymax']  = 0
  
  
  nx = im.shape[0]
  ny = im.shape[1]
  xmin = MAP['xmin']
  xresolution = 1
  ymin =  MAP['ymin']
  yresolution = 1
  nxs = xs.size
  nys = ys.size
  cpr = np.zeros((nxs, nys))
  #print(nys,nxs)
  #print(ny,nx,xs,ys)
  #print(vp[0])
  for jy in range(0,nys):
    y1 = vp[1,:] + ys[jy] # 1 x 1076
    iy = np.int16(np.round((y1-ymin)/yresolution))
    for jx in range(0,nxs):
      x1 = vp[0,:] + xs[jx] # 1 x 1076
      ix = np.int16(np.round((x1-xmin)/xresolution))
      valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
			                        np.logical_and((ix >=0), (ix < nx)))
      #print(valid)
      cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
  #print(cpr[0,0])
  return cpr


def bresenham2D(sx, sy, ex, ey):
  '''
  Bresenham's ray tracing algorithm in 2D.
  Inputs:
	  (sx, sy)	start point of ray
	  (ex, ey)	end point of ray
  '''
  sx = int(round(sx))
  sy = int(round(sy))
  ex = int(round(ex))
  ey = int(round(ey))
  dx = abs(ex-sx)
  dy = abs(ey-sy)
  steep = abs(dy)>abs(dx)
  if steep:
    dx,dy = dy,dx # swap 

  if dy == 0:
    q = np.zeros((dx+1,1))
  else:
    q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
  if steep:
    if sy <= ey:
      y = np.arange(sy,ey+1)
    else:
      y = np.arange(sy,ey-1,-1)
    if sx <= ex:
      x = sx + np.cumsum(q)
    else:
      x = sx - np.cumsum(q)
  else:
    if sx <= ex:
      x = np.arange(sx,ex+1)
    else:
      x = np.arange(sx,ex-1,-1)
    if sy <= ey:
      y = sy + np.cumsum(q)
    else:
      y = sy - np.cumsum(q)
  return np.vstack((x,y))
    

def test_bresenham2D():
  import time
  sx = 0
  sy = 1
  print("Testing bresenham2D...")
  r1 = bresenham2D(sx, sy, 10, 5)
  r1_ex = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10],[1,1,2,2,3,3,3,4,4,5,5]])
  r2 = bresenham2D(sx, sy, 9, 6)
  r2_ex = np.array([[0,1,2,3,4,5,6,7,8,9],[1,2,2,3,3,4,4,5,5,6]])	
  if np.logical_and(np.sum(r1 == r1_ex) == np.size(r1_ex),np.sum(r2 == r2_ex) == np.size(r2_ex)):
    print("...Test passed.")
  else:
    print("...Test failed.")

  # Timing for 1000 random rays
  num_rep = 1000
  start_time = time.time()
  for i in range(0,num_rep):
	  x,y = bresenham2D(sx, sy, 500, 200)
  print("1000 raytraces: --- %s seconds ---" % (time.time() - start_time))


def test_mapCorrelation():
  _, lidar_data = read_data_from_csv('data/sensor_data/lidar.csv')
  angles = np.linspace(-5, 185, 286) / 180 * np.pi
  ranges = lidar_data[0, :]

  # take valid indices
  indValid = np.logical_and((ranges < 80),(ranges> 0.1))
  ranges = ranges[indValid]
  angles = angles[indValid]

  # init MAP
  MAP = {}
  MAP['res']   = 0.1 #meters
  MAP['xmin']  = -50  #meters
  MAP['ymin']  = -50
  MAP['xmax']  =  50
  MAP['ymax']  =  50 
  MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
  MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
  MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
  
  #import pdb
  #pdb.set_trace()
  
  # xy position in the sensor frame
  xs0 = ranges*np.cos(angles)
  ys0 = ranges*np.sin(angles)
  
  # convert position in the map frame here 
  Y = np.stack((xs0,ys0))
  
  # convert from meters to cells
  xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
  yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
  
  # build an arbitrary map 
  indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
  MAP['map'][xis[indGood],yis[indGood]]=1
      
  x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
  y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map

  x_range = np.arange(-0.4,0.4+0.1,0.1)
  y_range = np.arange(-0.4,0.4+0.1,0.1)


  
  print("Testing map_correlation with {}x{} cells".format(MAP['sizex'],MAP['sizey']))
  ts = tic()
  c = mapCorrelation(MAP['map'],x_im,y_im,Y,x_range,y_range)
  print(c)
  toc(ts,"Map Correlation")

  c_ex = np.array([[ 4.,  6.,  6.,  5.,  8.,  6.,  3.,  2.,  0.],
                   [ 7.,  5., 11.,  8.,  5.,  8.,  5.,  4.,  2.],
                   [ 5.,  7., 11.,  8., 12.,  5.,  2.,  1.,  5.],
                   [ 6.,  8., 13., 66., 33.,  4.,  3.,  3.,  0.],
                   [ 5.,  9.,  9., 63., 55., 13.,  5.,  7.,  4.],
                   [ 1.,  1., 11., 15., 12., 13.,  6., 10.,  7.],
                   [ 2.,  5.,  7., 11.,  7.,  8.,  8.,  6.,  4.],
                   [ 3.,  6.,  9.,  8.,  7.,  7.,  4.,  4.,  3.],
                   [ 2.,  3.,  2.,  6.,  8.,  4.,  5.,  5.,  0.]])
    
  if np.sum(c==c_ex) == np.size(c_ex):
	  print("...Test passed.")
  else:
	  print("...Test failed. Close figures to continue tests.")	

  #plot original lidar points
  fig1 = plt.figure()
  plt.plot(xs0,ys0,'.k')
  plt.xlabel("x")
  plt.ylabel("y")
  plt.title("Laser reading")
  plt.axis('equal')

  #plot map
  fig2 = plt.figure()
  plt.imshow(MAP['map'],cmap="hot");
  plt.title("Occupancy grid map")
  
  #plot correlation
  fig3 = plt.figure()
  ax3 = fig3.gca(projection='3d')
  X, Y = np.meshgrid(np.arange(0,9), np.arange(0,9))
  ax3.plot_surface(X,Y,c,linewidth=0,cmap=plt.cm.jet, antialiased=False,rstride=1, cstride=1)
  plt.title("Correlation coefficient map")
  plt.show()
  
  
def show_lidar():
  _, lidar_data = read_data_from_csv('data/sensor_data/lidar.csv')
  angles = np.linspace(-5, 185, 286) / 180 * np.pi
  ranges = lidar_data[0, :]
  plt.figure()
  ax = plt.subplot(111, projection='polar')
  ax.plot(angles, ranges)
  ax.set_rmax(80)
  ax.set_rticks([0.5, 1, 1.5, 2])  # fewer radial ticks
  ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
  ax.grid(True)
  ax.set_title("Lidar scan data", va='bottom')
  plt.show()
  print(lidar_data[0,0])
  

def rotation_m(angles):
    x = angles[0]
    y = angles[1]
    z = angles[2]
    
    m_z = [[math.cos(z) , -math.sin(z) , 0],
           [math.sin(z) , math.cos(z)  , 0],
           [0           , 0            , 1]]
    
    m_y = [[math.cos(y) , 0 ,  math.sin(y)],
           [0           , 1            , 0],
           [-math.sin(y) , 0 , math.cos(y)]]
    
    m_x = [[1           , 0            , 0],
           [0 , math.cos(x) , -math.sin(x)],
           [0 , math.sin(x) , math.cos(x) ]]
    
    m_x = np.array(m_x)
    m_y = np.array(m_y)
    m_z = np.array(m_z)
    return np.dot(np.dot(m_z,m_y),m_x)

def get_v(pre_left_count,left_count, pre_right_count, right_count):
    return math.pi*(0.623479*(left_count - pre_left_count) + (right_count-pre_right_count) * 0.622806)/(4096*2)

def Velocity(path):
    timestamp,cnt = read_data_from_csv(path)
    l = cnt[:,0]
    r = cnt[:,1]
    d = collections.defaultdict(float)
    for i in range (1,timestamp.shape[0]):
        d[timestamp[i]//10**6] += get_v(l[i-1],l[i],r[i-1],r[i])
    return d,timestamp[1]
        
def W_velovity(path):
    timestamp,angles= read_data_from_csv(path)
    yaw = angles[:,2]
    d = collections.defaultdict(float)
    for i in range (1,timestamp.shape[0]):
        d[timestamp[i]//10**6] += yaw[i]
    return d

def get_FOG(path):
    timestamp,data= read_data_from_csv(path)
    d = collections.defaultdict(list)
    for i in range (0,timestamp.shape[0]):
        d[timestamp[i]//10**6] = data[i,:]
    return d

def get_robot_data():
    fogPath     = 'data/sensor_data/fog.csv'
    encoderPath = 'data/sensor_data/encoder.csv'
    lidarPath   = 'data/sensor_data/lidar.csv'
    d_v,begin_time = Velocity(encoderPath)
    d_w = W_velovity(fogPath)
    d_lidar = get_FOG(lidarPath)
    #d_intersect_vw = set(d_w.keys()).intersection(set(d_v.keys()))
    #print(set(d_w.keys()))
    d_new = collections.defaultdict(list)
    ans_w = 0
    ans_v = 0
    pretime = -1
    for time in d_w.keys():
        if time < begin_time//10**6:
            continue
        if time < pretime:
            print("Time Stamp Error")
            break;
        pretime = time
        ans_w += d_w[time]
        ans_v += d_v[time] 
        if time in d_v.keys() and time in d_lidar:
            d_new[time].append(ans_v)
            d_new[time].append(ans_w)
            d_new[time].append(d_lidar[time])
            ans_v = 0
            
    pos_x = [0]
    pos_y = [0]
    pre_yaw = 0
   
    for time in d_new.keys():
        delta_yaw = d_new[time][1] - pre_yaw
        pre_yaw = d_new[time][1]
        x = d_new[time][0]*math.cos(d_new[time][1])
        y = d_new[time][0]*math.sin(d_new[time][1])
        new_data = np.array([x,y])
        pos_x.append(pos_x[-1] + new_data[0])
        pos_y.append(pos_y[-1] + new_data[1]) 
        d_new[time][0] = new_data[0]
        d_new[time][1] = new_data[1]
        d_new[time].append(delta_yaw)
                                                 
    #print(len(d_new))
    plt.plot(pos_x,pos_y)
    plt.show()
    return  d_new


## initiallize particle position (random size, because do not know how manny points in the map) with corresponding posibility
## motion update  get the relative(x,y) position to zero point 
## weight update  (worldFrameLidarDate , map)
## compute the new probability of each particle 
## compute Neff about whether need to resample
## use delta pos to update particle's positioin at every timestamp

class Filter:
    def __init__(self,particleNumber = 1,Neff = 1.6):
        #self.pos_x = 
        #self.pos_y = 
        self.particleNumber = particleNumber
        self.particle = np.zeros((self.particleNumber,3))
        self.proba = np.array([1/self.particleNumber]*self.particleNumber)
        self.Neff  = Neff
        self.currentIteration = 0
        self.yaw = 0
        
        #init the map
        MAP = {}
        MAP['res']   = 1 #meters
        MAP['xmin']  = 0  #meters
        MAP['ymin']  = -1000-100
        MAP['xmax']  = 1200 + 100
        MAP['ymax']  = 0
        MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
        MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
        self.map      = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
        self.probaMap = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.float64) #DATA TYPE: char or int8
        
        
    def motion_update(self,update_data):
        # update the position of the particle using lidar data transformed  world frame
        #along with the noise
        # update_data formate [x,y,t]
        
        noiseToParticle = np.random.normal(0, 0, (self.particleNumber,3))
        self.particle += update_data
        self.particle += noiseToParticle
        
        return 
    
    def weight_update(self,bodyFrameLidarData):
        ## using correlationMap for update 
        ## need to transform the body frame lidar data to the world frame
        cr = np.zeros(shape=(self.particleNumber, 1))
        index = 0
        for particle in self.particle:
            rotation_matrix = rotation_m(np.array([0,0,particle[2]]))
            two_dim_rotation_m = rotation_matrix[0:2,0:2]
            
            particle_x = np.array([particle[0]])  # x
            particle_y = np.array([particle[1]])  # y
            world_frame_lidar_data = two_dim_rotation_m.dot(bodyFrameLidarData)
            #world_x = world_frame_lidar_data[0]
            #max_x = max(world_x)
            #min_x = min(world_x)
            #world_y = world_frame_lidar_data[1]
            #max_y = max(world_y)
            #min_y = min(world_y)
            
            c = mapCorrelation(self.map,[0,1300],[0,1100],world_frame_lidar_data,particle_x,particle_y)
            cr[index] = c
            index += 1
            
        s_cr = np.exp(cr-np.mean(cr))
        #s_cr = abs(cr-np.mean(cr)) + 1
        sum_cr = np.dot(self.proba , s_cr)
        #print(cr)
        for i in range(0,self.particleNumber):
            self.proba[i] = self.proba[i] * s_cr[i] / sum_cr
            #self.proba[i] *= cr[i]
            
        cur_neff = 1 / np.sum(np.square(self.proba))
        
        if cur_neff <= self.Neff:
            
            particle_list = np.random.choice(np.arange(0,self.particleNumber), self.particleNumber, p = self.proba) 
            # get self.particleNumber 个 sample in the self.particleNumber according to the self.proba
            self.particle = np.array(self.particle[particle_list])  ## double check here!!!!!!!!!!!!!!
            self.proba = np.array([1/self.particleNumber]*self.particleNumber)
            

        return
    
    
    def get_cur_iteration(self):
        return self.currentIteration
    
    def get_most_prob_pos(self):
        ## return the most posible position in the particle
        pos = np.where(self.proba == np.max(self.proba, axis=0)) # double check here
        #print(self.particle.shape)
        #print(self.particle[pos[0][0]])
        return random.choice(self.particle[pos[0]])
    
    def update_map(self,lidarScanData,minT = 0.1,maxT = 40):
        degrees = np.linspace(-5, 185, 286) / 180 * np.pi
        
        # take valid indices
        indValid = np.logical_and((lidarScanData < maxT), (lidarScanData > minT))
        valid_lidar = lidarScanData[indValid]
        degrees     = degrees[indValid]
        
        #lidar to world
        x_dis = np.array(valid_lidar * np.cos(degrees))
        y_dis = np.array(valid_lidar * np.sin(degrees))
        
        rotation_m_lidarToViecle = np.array([[0.00130201,0.796097, 0.605167],
                                             [0.999999, -0.000419027, -0.00160026],
                                             [-0.00102038, 0.605169, -0.796097]])
        
        rotation_m_lidarToViecle = rotation_m_lidarToViecle[0:2,0:2]
        
        #viecle to world
        new_dis = rotation_m_lidarToViecle.dot(np.array([x_dis,y_dis]))
        new_dis += np.array([[0.8349, -0.0126869]]).T
        
        pos   = self.get_most_prob_pos().reshape((3,-1))
        
        #viecle  to world
        rm = rotation_m(np.array([0,0,pos[2]]))
        new_dis_0  =  rm[:2,:2].dot(new_dis)
        #print("liarpos_x:  ",lidar_pos_x)
        
        lidar_pos_x = new_dis_0[0] + pos[0]
        lidar_pos_y = new_dis_0[1] + pos[1]
        
        posx_in_map = np.ceil((pos[0] - 0 )/1).astype(np.int16)
        posy_in_map = np.ceil((pos[1] + 1100)/1).astype(np.int16)
        
        if self.currentIteration % 10 == 0:
            self.weight_update(np.array([new_dis[0],new_dis[1]])) #update particle weights
        
        xis = np.ceil((lidar_pos_x - 0 ) / 1).astype(np.int16)
        yis = np.ceil((lidar_pos_y + 1100) / 1).astype(np.int16)
        
        indGood = np.logical_and(np.logical_and(np.logical_and((xis >=0 ), (yis >= 0)), (xis < 1300)),(yis < 1100))
        xis = xis[indGood]
        yis = yis[indGood]
        
        for index in range(0, len(xis)):
            x_point = xis[index]
            y_point = yis[index]
            stack = bresenham2D(posx_in_map[0], posy_in_map[0], x_point, y_point).astype(np.int16)
            
            if stack[0][-1] <=1300 and stack[1][-1]<= 1100:
            #print("+log4 position:  ",stack[1][-1]  ,stack[0][-1])
                self.probaMap[stack[0][-1]  ,stack[1][-1]]     += math.log(4)
            for j in range (0,len(stack[0])-1):
                if stack[1][j] <=1100 and stack[0][j]<= 1300:
                    self.probaMap[stack[0][j],stack[1][j]]     -= math.log(4)
            
        # 2 robot position  1 sturcture position 0 free position
        #valid = np.logical_and(self.probaMap > 0, self.map != 2)
        #print(posx_in_map,posy_in_map)
        self.map[np.logical_and(self.probaMap > 0, self.map != 2)] = 1
        self.map[np.logical_and(self.probaMap < 0, self.map != 2)] = 0
        #print("count1: ",collections.Counter(self.map.flatten()))
        if posx_in_map <=1300 and posy_in_map <= 1100:
            self.map[posx_in_map, posy_in_map] = 2
        

if __name__ == '__main__':
  #compute_stereo()
  #show_lidar()
  #test_mapCorrelation()
  #test_bresenham2D()
  #show_lidar()
  all_data = get_robot_data()  ##[delta_x delta_y scan_lidar delta_yaw]
  #d = get_FOG('data/sensor_data/lidar.csv') 
  #print(len(d))
  afilter = Filter()
  i = 0
  totallength =  len(all_data)
  for time in all_data.keys():
    #print("Time:  ",time)
    afilter.motion_update(np.array([all_data[time][0],all_data[time][1],all_data[time][3]]))  #motion update delta_x and delta_y and delta yaw
    afilter.update_map(np.array(all_data[time][2]))      ## update intakes lidarScanData
    afilter.currentIteration += 1
    i += 1
    if i%10000 == 0:
        #print(afilter.proba)
        print("progress: ", i/totallength*100 ,"%")
        plt.imshow(afilter.map)
        plt.show()
        print(np.sum(afilter.map),np.sum(afilter.probaMap))
        np.save('map_data_{}.npy'.format(i),afilter.map)
        np.save('mapprob_data_{}.npy'.format(i),afilter.probaMap)
       
  plt.imshow(afilter.map)
  plt.show()
  np.save('map_data.npy',afilter.map)
  np.save('mapprob_data.npy',afilter.probaMap)
      
