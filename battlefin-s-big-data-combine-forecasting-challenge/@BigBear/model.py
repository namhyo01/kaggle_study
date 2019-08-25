#open('last_observed_benchmark.csv','w').write("\n".join([",".join(['FileId']+['O'+str(o) for o in range(1,199)])] + [",".join([k]+open('data/'+k+'.csv').read().strip().split('\n')[-1].split(',')[:198]) for k in map(str,range(201,511))]))

import pickle
import numpy as np
import sys

from sklearn.cross_validation import StratifiedKFold
from sklearn import cross_validation
from multiprocessing import Pool
from sklearn import linear_model
n_folds = 6
folds = 7
SEED = 3
def MAE_loss( yy, pp ):
	err = np.mean(np.abs( yy - pp ) )
	return err

def cv_stack( model, X, ty, Xt ):
	sk = StratifiedKFold( ty, n_folds )
	ps = np.zeros( len(ty) )
	mean_err = 0
	for train_index, test_index in sk:
		X_train, X_test = X[ train_index ], X[ test_index ]
		y_train, y_test = ty[ train_index ], ty[ test_index ]
		model.fit( X_train, y_train )
		pp = model.predict( X_test )
		ps[ test_index ] = pp
		err = MAE_loss( y_test, pp )
		mean_err += err
	print 'Mean Error: %f' % (mean_err/n_folds)
	print 'Sum Error: %f' % MAE_loss( ty, ps )

	model.fit( X, ty )
	pred = model.predict( Xt )

	return (ps, pred)
#pickle.dump( (pall, preds), open(sub_fname, 'w') )
def cv1f( args1 ):
	X, y, index = args1
	random_seed = index*SEED + 6
	X_train, X_cv, y_train, y_cv= cross_validation.train_test_split( X, y, test_size=0.20,
			random_state = random_seed  )
	model.fit( X_train, y_train )
	preds = model.predict( X_cv )
	err = MAE_loss( y_cv, preds )
	return err

def cvsf( model, X, y ):
	mean_err = 0
	for i in range( folds ):
		err = cv1f( (X,y,i) )
		#print 'roc_auc: %f' % roc_auc
		mean_err += err
	#print 'mean_auc: %f ' % mean_auc
	return mean_err/folds

def cvpf( model, X, y ):
	pool = Pool(processes = folds)
	args = [ (X, y, i) for i in range( folds ) ]
	errors = pool.map( cv1f, args )
	pool.close()
	pool.join()
	return np.mean( errors )
def feature_score( arg ):
	feature_set, X, y = arg
	data = X[:, feature_set]
	return cvsf( model, data, y )

if sys.argv[1] == 'sub':
	fname = '%s.csv' % '_'.join( sys.argv[1:] )
	mf = 'res/%s.pkl' % sys.argv[2]
	y = pickle.load( open('output/y.pkl') )
	res = pickle.load( open( mf ) )
	ycv  = [ t[0] for t in res ]
	pp  = [ t[1] for t in res ]
	ycv  = np.hstack( np.column_stack( ycv ) )
	y    = np.hstack( y )
	pp = np.column_stack( pp )
	print pp.shape, y.shape, ycv.shape
	all_err = MAE_loss( y, ycv )
	with open(fname, 'w') as f:
		f.write('FileId,O1,O2,O3,O4,O5,O6,O7,O8,O9,O10,O11,O12,O13,O14,O15,O16,O17,O18,O19,O20,O21,O22,O23,O24,O25,O26,O27,O28,O29,O30,O31,O32,O33,O34,O35,O36,O37,O38,O39,O40,O41,O42,O43,O44,O45,O46,O47,O48,O49,O50,O51,O52,O53,O54,O55,O56,O57,O58,O59,O60,O61,O62,O63,O64,O65,O66,O67,O68,O69,O70,O71,O72,O73,O74,O75,O76,O77,O78,O79,O80,O81,O82,O83,O84,O85,O86,O87,O88,O89,O90,O91,O92,O93,O94,O95,O96,O97,O98,O99,O100,O101,O102,O103,O104,O105,O106,O107,O108,O109,O110,O111,O112,O113,O114,O115,O116,O117,O118,O119,O120,O121,O122,O123,O124,O125,O126,O127,O128,O129,O130,O131,O132,O133,O134,O135,O136,O137,O138,O139,O140,O141,O142,O143,O144,O145,O146,O147,O148,O149,O150,O151,O152,O153,O154,O155,O156,O157,O158,O159,O160,O161,O162,O163,O164,O165,O166,O167,O168,O169,O170,O171,O172,O173,O174,O175,O176,O177,O178,O179,O180,O181,O182,O183,O184,O185,O186,O187,O188,O189,O190,O191,O192,O193,O194,O195,O196,O197,O198\n')
		for ind, p in enumerate( pp ):
			f.write( str(ind + 201) )
			f.write( ',' )
			f.write( ','.join( str(x) for x in p ) )
			f.write('\n')
	print 'error : %f' % all_err
## load data
train = pickle.load( open('output/train.pkl') )

test = pickle.load( open('output/test.pkl') )

y = pickle.load( open('output/y.pkl') )

## create data from last observation
train_last = [ train[i][54, 0:199] for i in range(len(train)) ]
train_last = np.vstack( train_last )
test_last = [ test[i][54, 0:199] for i in range(len(test))]
test_last = np.vstack( test_last )

train2_last = [ train[i][54, :] for i in range( len(train) ) ]
train2_last = np.vstack( train2_last )
test2_last = [ test[i][54, :] for i in range( len(test) ) ]
test2_last = np.vstack( test2_last )

train3_last = [ train[i][51, :] for i in range( len(train) ) ]
train3_last = np.vstack( train3_last )
test3_last =  [ test[i][51, :] for i in range( len(test) ) ]
test3_last = np.vstack( test3_last )

ar_train, ar_test = pickle.load( open('output/ar.pkl') )

sub_fname = 'res/%s.pkl' % '_'.join( sys.argv[1:] )

if sys.argv[1] == 'linr':
	from linearMAE import LinearMAE
	from sklearn import preprocessing

	#ar_res = pickle.load( open('res/ar.pkl') )

	opt_method = 'bfgs'
	fs_method = 'rfs'
	if len(sys.argv) > 2:
		fs_method = sys.argv[2]
	elif fs_method == 'se':
		fea_lists = pickle.load( open( 'output/se_fs_gbm.pkl') )
	elif fs_method == 'fs':
		fea_lists = pickle.load( open( 'output/fs_gbm_50.pkl' ) )
	#fea_lists = pickle.load( open( 'output/%s_%s.pkl' % (fs_method, sys.argv[1])) )
	res = []
	np.random.seed( 42 )
	for Id, run_lists in enumerate( fea_lists ):
		print Id
		yy = y[ :, Id ]
		min_score, min_list = min(run_lists)
		train_last_fea = train2_last[:, min_list[-1] ]
		test_last_fea =test2_last[:, min_list[-1] ]

		yy = yy - train_last_fea

		trainX = train2_last[:, min_list]
		testX = test2_last[:, min_list]


		l1_ = 0.07
		l2_ = 0.07
		model = LinearMAE( l1 = l1_, l2 = l2_, opt= opt_method, verbose = True, maxfun = 1000 )
		ycv, pp = cv_stack( model, trainX, yy, testX )
		ycv = np.hstack( ycv )
		pp = np.hstack( pp )
		res.append( ( ycv  + train_last_fea , pp + test_last_fea ) )
	pickle.dump( res, open( sub_fname, 'w' ) )
if sys.argv[1] == 'ar':
	from linearCom import LinearCom
	from sklearn import preprocessing

	opt_method = 'bfgs'
	fea_lists = pickle.load( open( 'output/se_fs_gbm.pkl') )
	#linr_res = pickle.load( open('res/linr_se.pkl') )
	res = []
	for Id, run_lists in enumerate( fea_lists ):
		print Id
		yy = y[ :, Id ]
		min_score, min_list = min(run_lists)

		train_last_fea = train2_last[:, min_list[-1] ]
		test_last_fea =test2_last[:, min_list[-1] ]

		yy = yy - train_last_fea

		trainX = ar_train[Id][ :, -20: ]
		testX = ar_test[Id][ :, -20: ]

		l1_ = 0.07
		l2_ = 0.07
		model = LinearCom( l1 = l1_, l2 = l2_, opt= opt_method, verbose = True, maxfun = 1000 )
		ycv, pp = cv_stack( model, trainX, yy, testX )
		ycv = np.hstack( ycv )
		pp = np.hstack( pp )
		res.append( ( ycv  + train_last_fea , pp + test_last_fea ) )
	pickle.dump( res, open( sub_fname, 'w' ) )

if sys.argv[1] in ['gbm', 'rf']:
	from sklearn.ensemble import GradientBoostingRegressor
	from sklearn.ensemble import RandomForestRegressor

	fea_lists = pickle.load( open( 'output/se_fs_gbm2.pkl') )
	ar_res = pickle.load( open('res/ar.pkl') )
	#gbm_seed = int( sys.argv[2] )
	res = []
	for Id, run_lists in enumerate( fea_lists ):
		print Id
		yy = y[ :, Id ]
		min_score, min_list = min(run_lists)
		trainX = train2_last[:, min_list]
		testX = test2_last[:, min_list]

		train_last_fea = train2_last[:, -1]
		test_last_fea = test2_last[:, -1]
		
		yy = yy - train_last_fea

		scores = []
		for n_est in [ 50, 100 ]:
			for max_dp in [ 5, 7 ]:
				if sys.argv[1] == 'gbm':
					model = GradientBoostingRegressor( loss = 'lad', n_estimators = n_est,
							max_depth = max_dp )
				elif sys.argv[1] == 'rf':
					model = RandomForestRegressor( n_estimators = n_est, max_depth = max_dp )
				score = cvpf( model, trainX, yy )
				scores.append( (score, n_est, max_dp ) )
		score, n_est, max_dp = min( scores )
		print 'min score: %f' % score, n_est, max_dp
		if sys.argv[1] == 'gbm':
			model = GradientBoostingRegressor( loss = 'lad', n_estimators = n_est,
					max_depth = max_dp )
		elif sys.argv[1] == 'rf':
			model = RandomForestRegressor( n_estimators = n_est, max_depth = max_dp )
		ycv, pp = cv_stack( model, trainX, yy, testX )
		ycv = np.hstack( ycv )
		pp = np.hstack( pp )
		res.append( (ycv + train_last_fea, pp + test_last_fea) )
	pickle.dump( res, open( sub_fname, 'w' ) )

if sys.argv[1] == 'fs':
	from multiprocessing import Pool
	#do feature selection
	from linearMAE import LinearMAE
	from sklearn.ensemble import GradientBoostingRegressor
	import random

	secu_num = y.shape[1]
	feature_total = train2_last.shape[1]

	sysargv1 = sys.argv[1] 

	if sys.argv[2] == 'linr':
		model = LinearMAE( l1 = 0.1, l2 = 0.1, opt= 'bfgs', maxfun = 100 )
	if sys.argv[2] == 'gbm':
		model = GradientBoostingRegressor( loss = 'lad', n_estimators = 50,
				max_depth = 7 )

	if sysargv1 == 'fs':
		runs = 30
	feature_lists = []
	for i in range( secu_num ):
		run_list = []
		fea_list = [] #first use current security
		yy = y[:, i] #labels for current security

		train_last_fea = train2_last[:, -1]

		yy = yy - train_last_fea
		pool = Pool( processes = 20 )

		for run in range( runs ):
			test_features = [ [f] + fea_list for f in range( feature_total ) if f not in fea_list ]
			args = [ (feature_set, train2_last, yy ) for feature_set in test_features ]
			test_scores = []
			test_scores = pool.map( feature_score, args )

			score, feature_set = min( zip( test_scores, test_features ) )
			print 'Min Score: %f' % score 
			print 'feature set: ', feature_set
			fea_list = feature_set
			run_list.append( (score, fea_list) )
		pool.close()
		pool.join()
		feature_lists.append( run_list )
	pickle.dump( feature_lists, open('output/se_%s_%s.pkl' % ( sys.argv[1], sys.argv[2] ), 'w') )
