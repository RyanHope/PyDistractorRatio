__author__ = 'Ryan M. Hope <rmh3093@gmail.com>'

from random import choice, sample, uniform, gauss, random
from math import ceil, sqrt
import numpy as np
import array
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import string
from pybrain.utilities import asBinary

def generateObjects( r, present = 0, max = 48, rows = 6, cols = 8 ):
    """
    This generates a numpy matrix that holds all data about the objs in a
    distractor ratio experiment. The matrix has 'max' rows and 11 columns. The
    first column holds a numeric id for the object type (1='Red X',2='Green O',
    3='Red O',4='Green X'). The second and third column hold the row and column
    value of the object. The fourth and fifth column hold the x,y location
    of the object in visual angle with 0,0 being the dead center of the screen.
    The sixth column is 1 only for the target object, else the value is 0. The
    seventh and eighth columns are 1 only if the feature color or shape is
    available, else the value is 0. The ninth, tenth, and eleventh columns represent
    the activation, uncertainty and saliency scores of each object.
    
    NOTE: This function is really only tested for 48 objects where the display
    has 6 rows and 8 columns.
    """

    # Create the actual disctractor ratios based on a number index from 1-15
    ratio = [r * 3, max - r * 3]
    # Fill temp arrays with 1s and 2s corresponding the correct distractor ratio
    tmp1 = [1] * ratio[0]
    tmp2 = [2] * ratio[1]

    tid = 0
    # If target is present, make one of the objects in the temp arrays a target
    # by flipping one of its properties
    if ( present == 1 ):
        tid = 3
        tmp1[choice( tmp1 )] = tid
    elif ( present == 2 ):
        tid = 4
        tmp2[choice( tmp2 )] = tid

    tmp1.extend( tmp2 ) # Combine the two temp arrays into 1 array
    tmp1 = sample( tmp1, max ) # Shuffle the combined temp array

    # Create an empty matrix to hold all of the object data
    objs = np.empty( ( max, 11, ) )
    objs[:] = False

    # Calculate the row and column and x,y of each object
    for i in range( 0, max ):
        objs[i][0] = tmp1[i]
        objs[i][1] = ceil( i % cols ) + 1
        objs[i][2] = i / cols + 1
        objs[i][3] = objs[i][1] * 2 - cols + 1
        objs[i][4] = objs[i][2] * 2 - rows + 1
        if objs[i][0] == tid:
            objs[i][5] = 1
        else:
            objs[i][5] = 0

    return objs

def objASCII( obj ):
    """
    Returns an ASCII encoding of an object based on its numeric id. Lower case
    letter means the feature is not available, upper case letter means the feature
    is available.
    """
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

def objsASCII( objs, rows = 6, cols = 8 ):
    """
    Returns a string representation that when printed will result in a pretty
    ASCII representation of the trial objects.
    """

    symbols = []
    for obj in objs:
        symbols.append( objASCII( obj ) )

    ascii = ''
    for r in np.reshape( symbols, ( rows, cols ) ):
        for c in range( 0, len( r ) ):
            ascii = '%s%s, ' % ( ascii, r[c] )
        ascii = ascii[:-1] + '\n'
    return ascii[:-1]

def distance( x1, y1, x2, y2 ):
    """
    Calculate the distance between any two points.
    """
    return sqrt( ( x2 - x1 ) * ( x2 - x1 ) + ( y2 - y1 ) * ( y2 - y1 ) )

def quadratic_availability( ecc, size, coefvar, intercept, x_coeff, x2_coeff, fovea = 1 ):
    """
    Implements the quadratic available function from EPIC 4.
    """
    available = True
    if ( ecc > fovea ):
        if ( gauss( size, coefvar * size ) <= intercept + x_coeff * ecc + x2_coeff * ecc * ecc ):
            available = False
    return available

def color_availability( ecc, size, coeffofvar, x2_coeff ):
    """
    Color availability as in EPIC 4.
    """
    #print '   coefvar: %f   color: %f' % (coeffofvar,x2_coeff)
    return quadratic_availability( ecc, size, coeffofvar, .1, .1, x2_coeff )

def shape_availability( ecc, size, coeffofvar, x2_coeff ):
    """
    Shape availability as in EPIC 4.
    """
    #print '   coefvar: %f   shape: %f' % (coeffofvar,x2_coeff)
    return quadratic_availability( ecc, size, coeffofvar, .1, .1, x2_coeff )

def apply_availability( objs, fix, colorcoeff, shapecoeff, coeffofvar ):
    """
    Apply the color and shape availability functions to an objects maxtix.
    """
    for i in range( 0, len( objs ) ):
        if color_availability( distance( objs[i, 3], objs[i, 4], fix[0], fix[1] ), 1.8, coeffofvar, colorcoeff ):
            objs[i, 6] = True
        if shape_availability( distance( objs[i, 3], objs[i, 4], fix[0], fix[1] ), 1.8, coeffofvar, shapecoeff ):
            objs[i, 7] = True
    return objs

def make_all_visible( objs ):
    """
    Sets all the features of all objects in a objects matrix to be visible.
    """
    for i in range( 0, len( objs ) ):
        objs[i, 6] = True
        objs[i, 7] = True
    return objs

def get_target( objs ):
    """
    Returns the target object in a objects matrix if there is one.
    """
    for i in range( 0, len( objs ) ):
        if objs[i, 5]: return string.upper( objASCII( objs[i, ] ) )
    return None

def targetVisible( objs ):
    """
    Returns True if all of the features of the target object are visible, elsex = score_uncertainty(experiment.agent.objects,

    it returns False.
    """
    for i in range( 0, len( objs ) ):
        if objs[i, 5]:
            if objs[i, 6] and objs[i, 7]:
                return True
            else:
                return False
    return False

def score_salience( objs ):
    """
    Calculate the saliency score for each object in an objects matrix.
    """
    o = [objASCII( obj ) for obj in objs]
    nr = [r for r in o if r[0].upper() == 'R']
    nx = [x for x in o if x[1].upper() == 'X']
    color = ( 1.0 * len( nr ) ) / len( o )
    shape = ( 1.0 * len( nx ) ) / len( o )
    for i in range( 0, len( o ) ):
        if o[i].upper() == 'RX':
            objs[i, 10] = ( 1 - color ) + ( 1 - shape )
        elif o[i].upper() == 'RO':
            objs[i, 10] = ( 1 - color ) + shape
        if o[i].upper() == 'GX':
            objs[i, 10] = color + ( 1 - shape )
        elif o[i].upper() == 'GO':
            objs[i, 10] = color + shape
    return objs

def score_activation( objs, target, weights = ( 1, 1 ) ):
    """
    Calculate the activation score for each object in an objects matrix.
    """
    for i in range( 0, len( objs ) ):
        obj = objASCII( objs[i, ] )
        c = 0
        s = 0
        if ( obj[0] == target[0] ): c = weights[0]
        if ( obj[1] == target[1] ): s = weights[1]
        objs[i, 8] = c + s
    return objs

def score_uncertainty( objs, target, weights = ( 1, 1 ) ):
    """
    Calculate the uncertainty score for each object in an objects matrix.
    """
    for i in range( 0, len( objs ) ):
        obj = objASCII( objs[i, ] )
        c = weights[0]
        s = weights[1]
        if ( objs[i, 6] and obj[0] == target[0] ):
            c = weights[0] / 2
        elif ( objs[i, 6] and obj[0] != target[0] ):
            c = 0
        if ( objs[i, 7] and obj[1] == target[1] ):
            s = weights[1] / 2
        elif ( objs[i, 7] and obj[1] != target[1] ):
            s = 0
        objs[i, 9] = c + s
    return objs

def score( objs, target ):
    """
    Update the activation, uncertainty and saliency scores in an objects matrix.
    """
    objs = score_activation( objs, target )
    objs = score_uncertainty( objs, target )
    objs = score_salience( objs )
    return objs

def activation( x, y, fx, fy, a, s ):
    """
    The base of Pomplun's activation function which scores a pixel based on
    the density of the objects around it and those objects score.
    """
    return a * np.exp( -( ( ( fx - x ) * ( fx - x ) + ( fy - y ) * ( fy - y ) ) ) / ( 2 * s * s ) )

def pmap( x, y, objs, score = 9, s = 0.5 ):
    """
    Create a perceptual (topographic) map for a given score type.
    """
    map = sum( [activation( x, y, obj[3], obj[4], obj[score], s ) for obj in objs] )
    return map / np.max( map )

def detect_peaks( image ):
    """
    Detect the local maxima in a 2d matrix. Returns a 2d matrix where maxima are
    coded as 1s and the rest of the cells are 0s.
    """
    neighborhood = generate_binary_structure( 2, 2 )
    local_max = maximum_filter( image, footprint = neighborhood ) == image
    background = ( image == 0 )
    eroded_background = binary_erosion( background, structure = neighborhood, border_value = 1 )
    detected_peaks = local_max - eroded_background
    return detected_peaks

def get_maxima( map ):
    """
    Returns a vector of the local maxima in a perceptual map. Each cell in the
    vector is a 2-tuple where the first element is location of the peak and the
    second element is the value of the peak.
    """
    return [( p, map[p[0], p[1]] ) for p, v in np.ndenumerate( detect_peaks( map ) ) if v]

def get_highest( maxima, fix ):
    """
    Returns the highest maxima in a vector of maxima.
    """
    val = 0
    peak = fix
    for m in maxima:
        if m[1] > val:
            val = m[1]
            peak = m[0]
    return peak

def get_highest_val( maxima, fix ):
    """
    Returns the highest maxima value in a vector of maxima.
    """
    val = 0
    peak = fix
    for m in maxima:
        if m[1] > val:
            val = m[1]
            peak = m[0]
    return val

def get_nearest( maxima, fix ):
    """
    Returns the maxima nearest to the given fixation point a vector of maxima.
    """
    val = 99999
    peak = fix #None CWM 8 Feb 2012
    for m in maxima:
        d = distance( fix[0], fix[1], m[0][0], m[0][1] )
        if d < val and d > 0: #cwm added 'd > 0' on 8 Feb 2012
            val = d
            peak = m[0]
    return peak

def get_farthest( maxima, fix ):
    """
    Returns the maxima farthest from the given fixation point a vector of maxima.
    """
    val = 0 #-1 CWM 8 Feb 2012
    peak = fix
    for m in maxima:
        d = distance( fix[0], fix[1], m[0][0], m[0][1] )
        if d > val:
            val = d
            peak = m[0]
    return peak

###############################################################################

def sum_avail_features( objs ):
    sum_feats = 0.0
    for i in range( 48 ):
        #print '%i' % objs[i,9]
        sum_feats += objs[i, 9]
    return sum_feats


def all_feats_avail( objs ):
    item_feats = []
    for i in range( 48 ):
        item_feats.append( [( objs[i, 1], objs[i, 2] ), objs[i, 9]] )
    return item_feats

def get_all_colors( obs ):
    same_colors = 0.0
    for i in range( 48 ):
       if obs[i, 0] == 1 or obs[i, 0] == 3:
          same_colors = same_colors + 1
    return same_colors

def get_all_shapes( obs ):
    same_shapes = 0.0
    for i in range( 48 ):
       if obs[i, 0] == 2 or obs[i, 0] == 3:
          same_shapes = same_shapes + 1
    return same_shapes

def get_same_enc_colors( obs ):
    same_colors = 0.0
    for i in range( 48 ):
       if ( obs[i, 0] == 1 and obs[i, 6] == 1 ) or ( obs[i, 0] == 3 and obs[i, 6] == 1 ):
          same_colors = same_colors + 1
    return same_colors

def get_same_enc_shapes( obs ):
    same_shapes = 0.0
    for i in range( 48 ):
       if ( obs[i, 0] == 2 and obs[i, 7] == 1 ) or ( obs[i, 0] == 3 and obs[i, 7] == 1 ):
          same_shapes = same_shapes + 1
    return same_shapes
#========================================================================================
def get_all_enc_colors( obs ):
    all_colors = 0.0
    for i in range( 48 ):
       if obs[i, 6] == 1:
          all_colors = all_colors + 1
    return all_colors

def get_all_enc_shapes( obs ):
    all_shapes = 0.0
    for i in range( 48 ):
       if obs[i, 7] == 1:
          all_shapes = all_shapes + 1
    return all_shapes

def get_comp_items( obs ):
    all_obs = 0.0
    for i in range( 48 ):
       if ( obs[i, 6] == 1 and obs[i, 7] == 1 ):
          all_obs = all_obs + 1
    return all_obs
#========================================================================================


def get_fixated_obj_color( obs, nFixes, loc ):
    colorObs = nFixes
    for i in range( 48 ):
        if ( obs[i, 3] == loc[0] and obs[i, 4] == loc[1] ) and ( obs[i, 0] == 1 or obs[i, 0] == 3 ):
            colorObs = colorObs + 1
    return colorObs

def get_fixated_obj_shape( obs, nFixes, loc ):
    shapeObs = nFixes
    for i in range( 48 ):
        if ( obs[i, 3] == loc[0] and obs[i, 4] == loc[1] ) and ( obs[i, 0] == 2 or obs[i, 0] == 3 ):
            shapeObs = shapeObs + 1
    return shapeObs

def get_peak_count( map ):
    """
    Returns the the number of locations with a peak was detected using detect_peaks.
    """
    #val = 0
    peak_ct = 0
    peaks = detect_peaks( map )
    for r in range( 6 ):
        for c in range( 8 ):
            #print 'v = %f' % v[r,c]
            if peaks[r, c] == True:
                #print ' v = %f c = %f' % (v[r,c], criterion)
                peak_ct += 1
    return peak_ct

def get_rand_loc():
    loc = []
    x = sample_wr( [0, 1, 2, 3, 4, 5], 1 )
    y = sample_wr( [0, 1, 2, 3, 4, 5, 6, 7], 1 )
    loc.append( x[0] )
    loc.append( y[0] )
    return loc


def stop_absent( peak_type, nFixes, stop_crit ):
    '''
    Returns a boolean (True or False) regarding if the model should stop search and respond with target absent
    '''

    if nFixes >= stop_crit:
        response = True
    elif nFixes < stop_crit:
        response = False


    '''
    Nfeats = sum_avail_features(peak_type)
    if (Nfeats/96) > stop_crit:
        response = False
    elif (Nfeats/96) <= stop_crit:
        response = True
    '''
    '''
    Npeaks = get_peak_count(peak_type, peak_crit)
    Speaks = get_peak_sum(peak_type)
    print '   sum = %f' % Speaks
    '''
    '''
    Number of Peaks:
    if Npeaks > stop_crit:
        response = False
    elif Npeaks <= stop_crit:
        response = True
    '''
    '''
    Sum of Peaks:
    if Speaks > stop_crit:
        response = False
    elif Speaks <= stop_crit:
        response = True
    '''

    return response


def get_peak_sum( values ):
    """
    Returns the the totalsum of values.
    """
    #val = 0
    v = values
    totalsum = 0.0
    for r in range( 6 ):
        for c in range( 8 ):
            totalsum += v[r, c]
    return totalsum


def sample_wr( population, k ):
    "Chooses k random elements (with replacement) from a population"
    n = len( population )
    _random, _int = random, int  # speed hack random.random
    result = [None] * k
    for i in xrange( k ):
        j = _int( _random() * n )
        result[i] = population[j]
    return result

def selectAction():
    zero = 40 	#highest uncertainty
    one = 15  	#nearest uncertainty
    two = 15  	#farthest uncertainty
    three = 10	#highest salience
    four = 10	#nearest salience
    five = 10	#farthest salience	
    bag = []
    for i in range( 0, zero ):
	bag.append( 'zero' )
    for i in range( 0, one ):
	bag.append( 'one' )
    for i in range( 0, two ):
	bag.append( 'two' )
    for i in range( 0, three ):
	bag.append( 'three' )
    for i in range( 0, four ):
	bag.append( 'four' )
    for i in range( 0, five ):
	bag.append( 'five' )
    samp = sample_wr( bag, 1 )
    if samp[0] == 'zero':
	return 0
    elif samp[0] == 'one':
	return 1
    elif samp[0] == 'two':
   	return 2
    elif samp[0] == 'three':
	return 3
    elif samp[0] == 'four':
	return 4
    elif samp[0] == 'five':
	return 5

def getSaliencyData( filename, printcsv = False ):
	fin = open( filename )
	fin.readline() #PF
	wh = fin.readline().split( " " ) #width height
	width = int( wh[0] )
	height = int( wh[1] )
	#print width
	#print height
	line = fin.readline().strip()
        #outname = 'RawVals-' + filename +'.csv'
        #outfile = '/home/chris/Desktop/' + outname

	#fout = open(outfile, 'a')

	data = np.zeros( [width, height] )
	for h in range( height ):
		values = array.array( 'f' )
		#values = array('float32')
		values.read( fin, width )
		row = ""
                #fout.write('\n')
		for w in range( width ):
			if row != "": row = row + ","
			data[w][h] = values[w]
			v = str( values[w] )
                        #v = values[w]
			#fout.write('%.14f,' % v)
			if v == '0.0':
				v = '0'
			row = row + v

	fin.close()
        #fout.close() 
	return data
