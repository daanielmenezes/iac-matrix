from array import array
a = [ 1.0,2.0,3.0,\
      4.0,3.0,2.1,\
      1.0,2.5,1.5,\
      1.0,0.5,0.0 ]

b = [ 0.5,1.0,\
      5.0,2.0,\
      2.4,3.7 ]

f1 = open('floats_4x3_random.dat', 'wb')
f2 = open('floats_3x2_random.dat', 'wb')

f1_a = array('f', a)
f2_a = array('f', b)

f1_a.tofile(f1)
f2_a.tofile(f2)

f1.close()
f2.close()
