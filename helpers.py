from random import choice, sample, uniform, gauss
from math import ceil, sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import string
from pybrain.utilities import asBinary

obj_header = ("id","row","column","x","y","target","color","shape","s_activation","s_uncertainty")

def generateObjects(r, present=0):
    
    ratio = [r*3,48-r*3]
    tmp1 = [1]*ratio[0]
    tmp2 = [2]*ratio[1]
    
    tid = 0
    if (present==1):
        tid = 3
        tmp1[choice(tmp1)] = tid
    elif (present==2):
        tid = 4
        tmp2[choice(tmp2)] = tid
        
    tmp1.extend(tmp2)
    tmp1 = sample(tmp1,48)
    
    yoffset = 15.5/6
    xoffset = 15.5/8
    
    objs = np.empty((48,11,))
    objs[:] = False
    
    for i in range(0,48):
        objs[i][0] = tmp1[i]
        objs[i][1] = ceil(i%8) + 1 
        objs[i][2] = i/8 + 1
        objs[i][3] = objs[i][1]*xoffset - xoffset/2 + uniform(-.4,.4) - 7.75
        objs[i][4] = (objs[i][2]*yoffset - yoffset/2 + uniform(-.5,.5)) - 7.75
        if objs[i][0] == tid:
            objs[i][5] = 1
        else:
            objs[i][5] = 0
        
    return objs

def objASCII(obj):
    symbol = ''
    if obj[0] == 1:
        if obj[6]: symbol = symbol + 'R'
        else: symbol = symbol + 'r'
        if obj[7]: symbol = symbol + 'X'
        else: symbol = symbol + 'x'
    elif obj[0] == 2:
        if obj[6]: symbol = symbol + 'G'
        else: symbol = symbol + 'g'
        if obj[7]: symbol = symbol + 'O'
        else: symbol = symbol + 'o'
    elif obj[0] == 3:
        if obj[6]: symbol = symbol + 'R'
        else: symbol = symbol + 'r'
        if obj[7]: symbol = symbol + 'O'
        else: symbol = symbol + 'o'
    elif obj[0] == 4:
        if obj[6]: symbol = symbol + 'G'
        else: symbol = symbol + 'g'
        if obj[7]: symbol = symbol + 'X'
        else: symbol = symbol + 'x'
    return symbol    

def objsASCII(objs):
    
    symbols = []
    for obj in objs:
        symbols.append(objASCII(obj))
    
    ascii = ''   
    for r in np.reshape(symbols,(6,8)):
        for c in range(0,len(r)):
            ascii = '%s%s ' % (ascii,r[c])
        ascii = ascii[:-1] + '\n'
    return ascii[:-1]

def distance(x1,y1,x2,y2):
    return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))

def quadratic_availability(ecc, size, coefvar, intercept, x_coeff, x2_coeff, fovea=1):
    available = True
    if (ecc > fovea):
        if (gauss(size, coefvar*size) <= intercept + x_coeff * ecc + x2_coeff * ecc*ecc):
            available = False
    return available

def color_availability(ecc, size):
    return quadratic_availability(ecc,size,.7,.1,.1,.035)

def shape_availability(ecc, size):
    return quadratic_availability(ecc,size,.7,.1,.1,.3)

def apply_availability(objs,fx,fy):
    for i in range(0,len(objs)):
        if color_availability(distance(objs[i,3],objs[i,4],fx,fy),1.8):
            objs[i,6] = True
        if shape_availability(distance(objs[i,3],objs[i,4],fx,fy),1.8):
            objs[i,7] = True
    return objs

def get_target(objs):
    for i in range(0,len(objs)):
        if objs[i,5]: return string.upper(objASCII(objs[i,]))
    return None

def targetVisible(objs):
    for i in range(0,len(objs)):
        if objs[i,5]:
            if objs[i,6] and objs[i,6]:
                return True
            else:
                return False
    return False

def score_salience(objs):
    o = [objASCII(obj) for obj in objs]
    nr = [r for r in o if r[0].upper() == 'R']
    nx = [x for x in o if x[1].upper() == 'X']
    color = (1.0*len(nr))/len(o)
    shape = (1.0*len(nx))/len(o)
    for i in range(0,len(o)):
        if o[i].upper() == 'RX':
            objs[i,10] = (1-color) + (1-shape)
        elif o[i].upper() == 'RO':
            objs[i,10] = (1-color) + shape
        if o[i].upper() == 'GX':
            objs[i,10] = color + (1-shape)
        elif o[i].upper() == 'GO':
            objs[i,10] = color + shape
    return objs
    
def score_activation(objs,target,weights=(1,1)):
    for i in range(0,len(objs)):
        obj = objASCII(objs[i,])
        c = 0
        s = 0
        if (obj[0]==target[0]): c = weights[0]
        if (obj[1]==target[1]): s = weights[1]
        objs[i,8] = c + s
    return objs
    
def score_uncertainty(objs,target,weights=(1,1)):
    for i in range(0,len(objs)):
        obj = objASCII(objs[i,])
        c = weights[0]
        s = weights[1]
        if (objs[i,6] and obj[0]==target[0]):
            c = weights[0]/2
        elif (objs[i,6] and obj[0]!=target[0]):
            c = 0
        if (objs[i,7] and obj[1]==target[1]):
            c = weights[1]/2
        elif (objs[i,7] and obj[1]!=target[1]):
            c = 0    
        objs[i,9] = c + s
    return objs

def score(objs):
    target = get_target(objs)
    objs = score_activation(objs,target)
    objs = score_uncertainty(objs,target)
    return objs

def activation(x,y,fx,fy,a,s):
    return a * np.exp( -( ((fx-x)*(fx-x) + (fy-y)*(fy-y))) / (2*s*s) )

def pmap(x,y,objs,score=9,s=2):
    return sum([activation(x,y,obj[3],obj[4],obj[score],s) for obj in objs])

def detect_peaks(image):
    neighborhood = generate_binary_structure(2,2)
    local_max = maximum_filter(image, footprint=neighborhood)==image
    background = (image==0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks = local_max - eroded_background
    return detected_peaks

def frange(start,stop,step):
    return np.append(start,np.arange(start,stop,step)+step)

def get_maxima(map):
    return [p for p,v in np.ndenumerate(detect_peaks(map)) if v]

def index2fix(index):
    offset = 15.5/8
    a = ceil(index%8) + 1 
    b = index/8 + 1
    x = a*offset - offset/2 + uniform(-.4,.4) - 7.75
    y = b*offset - offset/2 + uniform(-.5,.5) - 7.75
    return (x,y[0])

def find_nearest(array,value):
    idx=(np.abs(array-value)).argmin()
    return idx,array[idx]

def bits(i,n): 
    return tuple((0,1)[i>>j & 1] for j in xrange(n-1,-1,-1))

def _encodeState(codes):
    return [int(''.join(map(str,code)),2) for code in codes]        

def encodeState(map, fixation):
    space = np.linspace(-7.75,7.75,8)
    fixation = [find_nearest(space,fixation[0])[0],find_nearest(space,fixation[1])[0]]
    peaks = detect_peaks(map)
    codes = []
    for r in range(0,len(peaks)):
        for c in range(0,len(peaks[r])):
            fix = 0
            if fixation[0] == c and fixation[1] == r:
                fix = 1
            code = bits(r,3) + bits(c,3) + bits(peaks[r,c],1) + bits(fix,1)
            #map(str, code)
            #state.append(int(''.join(map(str, code)),2))
            codes.append(code)
    return _encodeState(codes)