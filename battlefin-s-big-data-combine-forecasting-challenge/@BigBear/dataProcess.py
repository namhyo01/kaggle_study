#check one day, see how all the prices changes
import csv 
import pickle
import numpy as np

train = []
for dnum in range(1,201):
	if dnum == 22:
		pass
	else:
		day = []
		with open('data/%d.csv'% dnum ) as f:
			reader = csv.reader( f, delimiter=',' )
			reader.next()
			for row in reader:
				day.append( np.array( map( float, row ) ) )
		day = np.vstack( day )
		train.append( day )

test = []
for dnum in range(201, 511):
	day = []
	with open('data/%d.csv'% dnum ) as f:
		reader = csv.reader( f, delimiter=',' )
		reader.next()
		for row in reader:
			day.append( np.array( map( float, row ) ) )
	day = np.vstack( day )
	test.append( day )
pickle.dump( train, open('output/train.pkl', 'w') )
pickle.dump( test, open( 'output/test.pkl', 'w' ) )

y = []
with open( 'trainLabels.csv' ) as f:
	reader = csv.reader( f, delimiter=',' ) 
	reader.next()
	count = 1
	for row in reader:
		if count == 22:
			pass
		else:
			y.append( np.array( map( float, row ) )[1:] )
		count += 1
y = np.vstack( y )
pickle.dump( y, open('output/y.pkl', 'w' ) )


train_ar = []
test_ar = []
for i in range( 198 ):
	ar_train = []
	ar_test = []
	for j in range(len(train)):
		ar_train.append( train[j][:,i] )
	for j in range(len(test)):
		ar_test.append( test[j][:,i] )
	ar_train = np.vstack( ar_train )
	ar_test = np.vstack( ar_test )
	train_ar.append( ar_train )
	test_ar.append( ar_test )

pickle.dump( (train_ar,test_ar), open('output/ar.pkl', 'w' ) )

train_diff = []
test_diff = []
for i in range( 198 ):
	diff_1 = []
	diff_2 = []
	for j in range( len(train)):
		diff_day = train[j][1:,i]
		for k in reversed( range( 1, len(diff_day) ) ):
			diff_day[ k ] = diff_day[k] - diff_day[k-1]
		diff_1.append( diff_day )
	for j in range( len(test) ):
		diff_day = test[j][1:,i]
		for k in reversed( range( 1, len(diff_day) ) ):
			diff_day[ k ] = diff_day[k] - diff_day[k-1]
		diff_2.append( diff_day )
	diff_1 = np.vstack( diff_1 )
	diff_2 = np.vstack( diff_2 )
	train_diff.append( diff_1 )
	test_diff.append( diff_2 )
pickle.dump( (train_diff, test_diff), open('output/diff.pkl', 'w') )


inst_len = 198 
out_len = 244

'''
from pylab import *

day1 = train[0]
x = range( 0, day1.shape[0] )
for t in range( inst_len + 1 ):
	plot( x, day1[:, t] )
show()
'''
