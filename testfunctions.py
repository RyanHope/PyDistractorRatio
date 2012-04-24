import random

def sample_wr( population, k ):
    "Chooses k random elements (with replacement) from a population"
    n = len( population )
    _random, _int = random.random, int  # speed hack 
    result = [None] * k
    for i in xrange( k ):
        j = _int( _random() * n )
        result[i] = population[j]
    return result

def selectAction():
    one = 50
    two = 10
    three = 10
    four = 10
    five = 10
    six = 10
    bag = []
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
    for i in range( 0, six ):
	bag.append( 'six' )
    samp = sample_wr( bag, 1 )
    if samp[0] == 'one':
	return 1
    elif samp[0] == 'two':
	return 2
    elif samp[0] == 'three':
   	return 3
    elif samp[0] == 'four':
	return 4
    elif samp[0] == 'five':
	return 5
    elif samp[0] == 'six':
	return 6

def testSelectAction( n ):
    f = open( '/home/chris/Desktop/testdata', 'w' )
    for i in range( 0, n ):
      a = getAction()
      f.write( '%d\n' % ( a ) )
      print a
    f.close()

