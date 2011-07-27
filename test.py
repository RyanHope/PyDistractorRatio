import numpy as np
from helpers import *

objs = generateObjects(7,1)
x,y = np.meshgrid(np.linspace(-7.75,7.75,8),np.linspace(-7.75,7.75,8))

objs = apply_availability(objs,0,0)
objs = score(objs)
print objsASCII(objs)
z = pmap(x,y,objs,score=8)
plt.subplot(2,5,1)
plt.imshow(z)
z = pmap(x,y,objs,score=9)
plt.subplot(2,5,6)
plt.imshow(z)

objs = apply_availability(objs,-7.75,-7.75)
objs = score(objs)
print objsASCII(objs)
z = pmap(x,y,objs,score=8)
plt.subplot(2,5,2)
plt.imshow(z)
z = pmap(x,y,objs,score=9)
plt.subplot(2,5,7)
plt.imshow(z)

objs = apply_availability(objs,-7.75,7.75)
objs = score(objs)
print objsASCII(objs)
z = pmap(x,y,objs,score=8)
plt.subplot(2,5,3)
plt.imshow(z)
z = pmap(x,y,objs,score=9)
plt.subplot(2,5,8)
plt.imshow(z)

objs = apply_availability(objs,7.75,7.75)
objs = score(objs)
print objsASCII(objs)
z = pmap(x,y,objs,score=8)
plt.subplot(2,5,4)
plt.imshow(z)
z = pmap(x,y,objs,score=9)
plt.subplot(2,5,9)
plt.imshow(z)

objs = apply_availability(objs,7.75,-7.75)
objs = score(objs)
print objsASCII(objs)
z = pmap(x,y,objs,score=8)
plt.subplot(2,5,5)
plt.imshow(z)
z = pmap(x,y,objs,score=9)
plt.subplot(2,5,10)
plt.imshow(z)








fig = plt.figure()
ax = fig.gca(projection='3d')
cset = ax.contour(x,y,z,16,extend3d=True)
ax.clabel(cset, fontsize=9, inline=1) 
plt.show()


print objsASCII(objs)
plt.matshow(perception.pmap(frange(-7.75,7.75,.1),frange(-7.75,7.75,.1),objs[:,3],objs[:,4],objs[:,9],2))




objs = generateObjects(5,1)
objs = apply_availability(objs,0,0)
target = get_target(objs)


objs = apply_availability(objs,0,7)
target = get_target(objs)
objs = score_activation(objs,target)
objs = score_uncertainty(objs,target)




#plt.scatter(objs[:,3], objs[:,4])





objs = generateObjects(7,1)

objs = apply_availability(objs,-7,-7)
objs = score(objs)
z1 = pmap(x,y,objs,score=8)
z1 = z1 / np.max(z1)
z2 = pmap(x,y,objs,score=9)
z2 = z2 / np.max(z2)
z3 = z1*z2
plt.subplot(1,3,1)
plt.imshow(z1)
plt.subplot(1,3,2)
plt.imshow(z2)
plt.subplot(1,3,3)
plt.imshow(z3)
