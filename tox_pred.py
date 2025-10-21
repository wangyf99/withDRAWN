import random
import math
import numpy as np
import pandas as pd
import statistics
from sklearn.ensemble import RandomForestClassifier
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import GenericUnivariateSelect, chi2

def load_labels(label_key_number, random_seed, train1_percent, train2_percent, test_percent):
    '''loads drugs corresponding to a ballanced training and two test sets

    Input:
    label_key_number (int)- column in the label file to use as the labels
    random_seed (int) - controls the randomness for replicability if desired
    train_percent (float) - percent of the set used for testing, recomend 0.9 or less

    Output: (train, test1, test2, Ltrain, Ltest1, Ltest2) (tuple of lists) lists of labels and lists of drug names corresponding to a ballanced training and two test sets
    '''
    positive = []
    negative = []
    with open('tox_labels.csv') as fo:
        i = 0 
        for line in fo:
            if i != 0: 
                split_line = line[:-1].split(',')
                if split_line[label_key_number] == '1':
                    positive.append(split_line[0].lower())
                else:
                    negative.append(split_line[0].lower())
            i += 1

    random.seed(random_seed)
    
    size_samples = math.floor(min(len(positive)*(train1_percent), len(negative)*(train1_percent)))
    size_samples2 = math.floor(min(len(positive)*(train2_percent), len(negative)*(train2_percent)))
    size_test = math.floor(min(len(positive)*(test_percent), len(negative)*(test_percent)))

    rearrange_positive = random.sample(positive, len(positive))
    rearrange_negative = random.sample(negative, len(negative))

    train1 = rearrange_positive[:size_samples] + rearrange_negative[:size_samples] 
    train2 = rearrange_positive[size_samples:size_samples+size_samples2] + rearrange_negative[size_samples:size_samples+size_samples2] 
    test1 = rearrange_positive[size_samples+size_samples2:size_samples+size_samples2+size_test] + rearrange_negative[size_samples+size_samples2:size_samples+size_samples2+size_test]
    
    Ltrain1 = [1]*len(rearrange_positive[:size_samples]) + [0]*len(rearrange_negative[:size_samples])
    Ltrain2 = [1]*len(rearrange_positive[size_samples:size_samples+size_samples2]) + [0]*len(rearrange_negative[size_samples:size_samples+size_samples2])
    Ltest1 = [1]*len(rearrange_positive[size_samples+size_samples2:size_samples+size_samples2+size_test]) + [0]*len(rearrange_negative[size_samples+size_samples2:size_samples+size_samples2+size_test])
    return train1, train2, test1, Ltrain1, Ltrain2, Ltest1

    #9010
    # size_samples = math.floor(min(len(positive)*(train_percent), len(negative)*(train_percent)))
    # size_test = math.floor(min(len(positive)*(1-train_percent), len(negative)*(1-train_percent)))

    # rearrange_positive = random.sample(positive, len(positive))
    # rearrange_negative = random.sample(negative, len(negative))

    # train1 = rearrange_positive[:size_samples] + rearrange_negative[:size_samples] 
    # test1 = rearrange_positive[size_samples:size_samples+size_test] + rearrange_negative[size_samples:size_samples+size_test]
    # Ltrain1 = [1]*len(rearrange_positive[:size_samples]) + [0]*len(rearrange_negative[:size_samples])
    # Ltest1 = [1]*len(rearrange_positive[size_samples:size_samples+size_test]) + [0]*len(rearrange_negative[size_samples:size_samples+size_test])
    # return train1, train1, test1, Ltrain1, Ltrain1, Ltest1

    #454510
    # size_samples = math.floor(min((len(positive)*(train_percent))/2, (len(negative)*(train_percent))/2))
    # size_test = math.floor(min(len(positive)*(1-train_percent), len(negative)*(1-train_percent)))

    # rearrange_positive = random.sample(positive, len(positive))
    # rearrange_negative = random.sample(negative, len(negative))

    # train1 = rearrange_positive[:size_samples] + rearrange_negative[:size_samples] 
    # train2 = rearrange_positive[size_samples:size_samples*2] + rearrange_negative[size_samples:size_samples*2] 
    # test1 = rearrange_positive[size_samples*2:size_samples*2+size_test] + rearrange_negative[size_samples*2:size_samples*2+size_test]
    # Ltrain1 = [1]*len(rearrange_positive[:size_samples]) + [0]*len(rearrange_negative[:size_samples])
    # Ltrain2 = [1]*len(rearrange_positive[size_samples:size_samples*2]) + [0]*len(rearrange_negative[size_samples:size_samples*2])
    # Ltest1 = [1]*len(rearrange_positive[size_samples*2:size_samples*2+size_test]) + [0]*len(rearrange_negative[size_samples*2:size_samples*2+size_test])

    #return train1, train2, test1, Ltrain1, Ltrain2, Ltest1

def load_nongraph(drug_names_list, filename):
    ''' load data for the drugs in the list from the file
    
    Inputs:
    drug_names_list (list of str)
    filename (str)

    Output: (list of lists) drug data matrix
    '''
    track = {}
    with open(filename) as fo:
        for line in fo:
            split_line = line[:-1].split(',')
            convert_to_float = []
            for elt in split_line[1:]:
                try:
                    convert_to_float.append(float(elt))
                except:
                    convert_to_float.append(0)
            track[split_line[0].lower()] = convert_to_float
    out = []
    for drug in drug_names_list:
        out.append(track[drug])
    return out

def load_data(drug_names_list):
    ''' load drug data for all predictors

    Input:
    drug_names_list (list of str)

    Output: (sages_out, fp_out, drug_features_out, targetsall) (tuple of pandas data frames) datasets for each of the predictors
    '''
    sages_out = pd.DataFrame(load_nongraph(drug_names_list, 'sages.csv'))
    fp_out = pd.DataFrame(load_nongraph(drug_names_list, 'fp.csv'))
    drug_features_out = pd.DataFrame(load_nongraph(drug_names_list, 'drug_features.csv'))
    targetsall = pd.DataFrame(load_nongraph(drug_names_list, 'targetsall.csv'))
    return sages_out, fp_out, drug_features_out, targetsall

def norm_data_by_train(trainset, testset1):
    '''minmax normalizes the data by the traiining set

    Inputs:
    trainset (pandas data frame)
    testset1 (pandas data frame)

    Output: normalized pandas data frame of the testset 
    '''
    mins = trainset.min(axis=0)
    maxs = trainset.max(axis=0) 
    for colnumber in range(testset1.shape[1]):
        mi = mins[colnumber]
        ma = maxs[colnumber]
        testset1[colnumber] = (testset1[colnumber]-mi)/(ma-mi)
    return testset1.fillna(0)

def evaluate(y_test, y_test_predict):
    '''returns performance metrics for machine learning classifiers

    Inputs:
    y_test (list) true values of the labels
    y_test_predict (list) predicted values of the labels output from the classifier

    Output: acc,aroc,f1_val,precision_val,recall_val,mcc (tupl of floats) performance metric values
    '''
    acc = accuracy_score(y_test, y_test_predict)
    aroc = roc_auc_score(y_test, y_test_predict)
    f1_val = f1_score(y_test, y_test_predict)
    precision_val = precision_score(y_test, y_test_predict)
    recall_val = recall_score(y_test, y_test_predict)
    mcc = matthews_corrcoef(y_test, y_test_predict)
    return acc,aroc,f1_val,precision_val,recall_val,mcc

def split_norm_data(train, test):
    ''' loads the data and normalizes by the training set

    Inputs: train, test1, test2 each (list) contains names of the drugs (str) in the data subset 

    Output:
    list of tuples where the first index of the tuple is the dataset label (str) and the remainder are pandas dataframes corresponding to train_set,test_set1,test_set2
    '''
    load_train = load_data(train)
    load_test = load_data(test) 
    l = ['sages', 'fp', 'drug_features', 'targetsall']
    out = []
    for data_i in range(len(load_train)):
        train_set = norm_data_by_train(load_train[data_i],load_train[data_i])
        test_set = norm_data_by_train(load_train[data_i],load_test[data_i])
        out.append((l[data_i], train_set, test_set))
    return out

def tuning_level1(label_key_number, random_seed, num_test_itterations, train1_percent, train2_percent, test_percent, classifiers, cl, outdir, write=True):
    '''model selection and hyperparameter tuning for each of the datasets

    Inputs:
    label_key_number (int) column number corresponding to which labels to use in the tox_labels.csv file
    random_seed (int) for model instantiation and dataset splitting
    train_percent (float) amount of the dataset used for training, the remaning data will be split into two test sets
    classifiers (list of TPOT classifiers)
    cl (list of str) classifier labels with the same indexing as classifiers
    outdir (str) directory where output files will be saved
    write (boolean) for debugging purposes, False will prevent any files from being saved
    
    Outputs: returns None, but will save the following files (if the write variable is True)
    classifier name - dataset -level1-tpot_exported_pipeline.py: a python file with the best hyperparameter tuned classifer
    dataset -level1_out_train_labels.csv: labels for the training data set for the ensemble
    dataset -level1_out_test_labels.csv: labels for the test set for the ensemble
    level1_summary.csv: performance metrics for all classifiers trained on all datasets
    dataset - classifier - random seed -level2_train.csv: predictions for the classifier on the data set, used as training data for the ensemble
    dataset - classifier - random seed -level2_test.csv: predictions for the classifier on the data set, used as testing data for the ensemble
    '''
    train1, train2, test1, Ltrain1, Ltrain2, Ltest1 = load_labels(label_key_number, random_seed, train1_percent, train2_percent, test_percent)
    data = split_norm_data(train1, test1)
    if write:
        fout0 = open(outdir+'level1_summary.csv', '+a')
        fout0.write('RandomSeed,Data,Accuracy,AUROC,F1,Precision,Recall,MCC,Classifier\n')
        fout0.close()
    
    for data_set_i in range(len(data)):
        print('****************')
        data_set = data[data_set_i]
        print(data_set[0])
        train_set = data_set[1]
        # test_set1 = data_set[2]
        # test_set2 = data_set[3]
        
        for clf_i in range(len(classifiers)):
            clf = classifiers[clf_i]
            clf.fit(train_set,np.array(Ltrain1))
            exctracted_best_model = clf.fitted_pipeline_.steps[-1][1]
            if write:
                clf.export(outdir+cl[clf_i]+ '-' + data_set[0]+'-level1-tpot_exported_pipeline.py')
            for rs in range(random_seed, random_seed+num_test_itterations):
                rstrain1, rstrain2, rstest1, rsLtrain1, rsLtrain2, rsLtest1 = load_labels(label_key_number, rs, train1_percent, train2_percent, test_percent)
                
                rsdata = split_norm_data(rstrain1, rstest1)
                rsdata2 = split_norm_data(rstrain2, rstest1)
                rstrain_set1 = rsdata[data_set_i][1]
                rstest_set1 = rsdata[data_set_i][2]

                rstrain_set2 = rsdata2[data_set_i][1]
                rstest_set2 = rsdata2[data_set_i][2]

                rsmodel = exctracted_best_model.fit(rstrain_set1,np.array(rsLtrain1))

                y_predict_test1 = rsmodel.predict(rstest_set1)
                
                acc,aroc,f1_val,precision_val,recall_val,mcc = evaluate(np.array(rsLtest1), y_predict_test1)
                
                if write:
                    fout1 = open( outdir + data_set[0]+'-level1_out_train_labels.csv','a+')
                    for elt in rsLtrain2:
                        fout1.write(str(elt)+ ',')
                    fout1.write('\n')
                    fout1.close()
                    fout2 = open(outdir + data_set[0]+'-level1_out_test_labels.csv' ,'a+')
                    for elt in rsLtest1:
                        fout2.write(str(elt)+ ',')
                    fout2.write('\n')
                    fout2.close()
                    out_str = str(rs)+','+data_set[0]+','+str(acc)+','+str(aroc)+','+str(f1_val)+','+str(precision_val)+','+str(recall_val) +','+str(mcc)+','+ cl[clf_i] +'\n'
                    fout0 = open(outdir+'level1_summary.csv', '+a')
                    fout0.write(out_str)
                    fout0.close()

                    y_predict_test2 = rsmodel.predict_proba(rstest_set2)
                    y_predict_train2 = rsmodel.predict_proba(rstrain_set2)
                    fout3 = open( outdir + data_set[0]+'-'+cl[clf_i]+'-'+str(rs)+'-level2_train.csv','a+')
                    for elt in list(y_predict_train2):
                        fout3.write(str(elt[0])+ ',')
                    fout3.write('\n')
                    fout3.close()
                    fout4 = open( outdir+ data_set[0]+'-'+ cl[clf_i]+'-'+str(rs)+'-level2_test.csv','a+')
                    for elt in list(y_predict_test2):
                        fout4.write(str(elt[0])+ ',')
                    fout4.write('\n')
                    fout4.close()

def get_label_from_l1(outdir, train_or_test):
    '''reads the file containing labels of the training and test sets for the ensemble

    Inputs:
    outdir (str) directory where the file containing labels of the training and test sets can be found
    train_or_test (str) either 'train' or 'test'

    Outputs:
    list of 1 and 0s corresponding to the labels of the training and test sets for the ensemble
    '''
    with open(outdir + 'sages-level1_out_'+ train_or_test+'_labels.csv') as fo:
        for line in fo:
            split_line = line.replace('\n','').split(',')
            out = []
            for elt in split_line[:-1]:
                out.append(float(elt))
            return out

def tuning_level2(classifiers, cl, outdir, write = True):
    '''trains the ensemble and evaluates

    Inputs:
    classifiers (list of TPOT classifiers)
    cl (list of str) classifier labels with the same indexing as classifiers
    outdir (str) directory where output files will be saved
    write (boolean) for debugging purposes, False will prevent any files from being saved

    Outputs:returns None, but will save the following files (if the write variable is True)
    classifier name - dataset -level2-tpot_exported_pipeline.py: a python file with the best hyperparameter tuned classifer
    level2_summary.csv: performance metrics for all classifiers trained
    '''
    train_data = pd.read_csv(outdir+'0-level2_train.csv', header=None)
    train_data = train_data.transpose()
    train_data.drop(train_data.tail(1).index,inplace=True)
    Ltrain = get_label_from_l1(outdir, 'train')
    if write:
        fout0 = open(outdir+'level2_summary.csv', '+a')
        fout0.write('RandomSeed,Accuracy,AUROC,F1,Precision,Recall,MCC,Classifier\n')
        fout0.close()

    for clf_i in range(len(classifiers)):
        clf = classifiers[clf_i]
        clf.fit(train_data,np.array(Ltrain))
        exctracted_best_model = clf.fitted_pipeline_.steps[-1][1]
        if write:
            clf.export(outdir+cl[clf_i]+ '-level2-tpot_exported_pipeline.py')

        for rs in range(10):
            train_data = pd.read_csv(outdir+str(rs)+'-level2_train.csv', header=None)
            train_data = train_data.transpose()
            test_data = pd.read_csv(outdir+str(rs)+'-level2_test.csv', header=None)
            test_data = test_data.transpose()
            train_data.drop(train_data.tail(1).index,inplace=True)
            test_data.drop(test_data.tail(1).index,inplace=True)
            Ltrain = get_label_from_l1(outdir, 'train')
            Ltest = get_label_from_l1(outdir, 'test')
            rsmodel = exctracted_best_model.fit(train_data,np.array(Ltrain)) 
            y_predict_test1 = rsmodel.predict(test_data)
            acc,aroc,f1_val,precision_val,recall_val,mcc = evaluate(np.array(Ltest), y_predict_test1)
            if write:
                out_str = str(rs)+','+str(acc)+','+str(aroc)+','+str(f1_val)+','+str(precision_val)+','+str(recall_val) +','+str(mcc) +','+ cl[clf_i] +'\n'
                fout0 = open(outdir+'level2_summary.csv', '+a')
                fout0.write(out_str)
                fout0.close()

def level1_fs2(label_key, outdir):
    '''feature selection for the best performing classifiers trained on each of the datasets 

    Inputs:
    label_key (int) column number corresponding to which labels to use in the tox_labels.csv file
    outdir (str) directory where output files will be saved

    Outputs: returns None, but will save the following files
    dataset _variancefs.csv feature selection according to the variance threshold
    dataset _genericunifs.csv feature selection using Generic Univariate Selection
    dataset _fs.csv feature seleciton using recursive feature elimination with cross-validation using a random forest classifier and a random seed of 0
    '''
    l = {'sages':{}, 'target':{}, 'fp':{}, 'drug_features':{}}
    for rs in range(10):
        print(rs)
        train, test1, test2, Ltrain, Ltest1, Ltest2 = load_labels(label_key, rs, 0.8,0.1,0.1)
        data = split_norm_data(train, test1, test2)
    
        for data_set_i in range(len(data)):
            data_set = data[data_set_i]
            set_name = data_set[0]
            if set_name in ['sages','targetsall', 'drug_features']:
                print(set_name)
                train_set = data_set[1]
                
                selector = VarianceThreshold()
                selector.fit(train_set)
                fout = open(outdir+set_name+'_variancefs.csv', 'a+' )
                for elt in selector.get_support(indices=False):
                    fout.write(str(elt)+ ',')
                fout.write('\n')
                fout.close()

                transformer = GenericUnivariateSelect(chi2, mode='k_best', param=int(train_set.shape[1]/4))
                transformer.fit(train_set, np.array(Ltrain))
                fout = open(outdir+set_name+'_genericunifs.csv', 'a+' )
                for elt in transformer.get_support(indices=False):
                    fout.write(str(elt) + ',')
                fout.write('\n')
                fout.close()

                selector = RFECV(RandomForestClassifier(random_state=rs), step=1, cv=5)
                selector = selector.fit(train_set, np.array(Ltrain))
                fout = open(outdir+set_name+'_fs.csv', 'a+' )
                for elt in selector.ranking_:
                    fout.write(str(elt) + ',')
                fout.write('\n')
                fout.close()

def get_prroc(prroc_conds, label_key_number, random_seed, train1_percent, train2_percent, test_percent,classifiers, cl, outdir, write=True):
    '''calculates the precision recall and receiver operator characteristic curves 

    Inputs:
    prroc_conds (list of str) conatins a list of best classifiers for each dataset 
    label_key_number (int) column number corresponding to which labels to use in the tox_labels.csv file
    random_seed (int) for model instantiation and dataset splitting
    train_percent (float) amount of the dataset used for training, the remaning data will be split into two test sets
    classifiers (list of TPOT classifiers)
    cl (list of str) classifier labels with the same indexing as classifiers
    outdir (str) directory where output files will be saved
    write (boolean) for debugging purposes, False will prevent any files from being saved

    Outputs: returns None, but will save the following files (if the write variable is True)
    random seed dataset - classifier -curves.csv file containing false positive rate, true positive rate, precision and recall for classifier traind on the random seed
    '''
    # train, test1, test2, Ltrain, Ltest1, Ltest2 = load_labels(label_key_number, random_seed, train1_percent, train2_percent, test_percent)
    # data = split_norm_data(train, test1, test2)
    train1, train2, test1, Ltrain1, Ltrain2, Ltest1 = load_labels(label_key_number, random_seed, train1_percent, train2_percent, test_percent)
    data = split_norm_data(train1, test1)
    
    for data_set_i in range(len(data)):
        print('****************')
        data_set = data[data_set_i]
        print(data_set[0])
        train_set = data_set[1]
        # test_set1 = data_set[2]
        # test_set2 = data_set[3]
        
        for clf_i in range(len(classifiers)):
            clf = classifiers[clf_i]
            if data_set[0]+ '-' + cl[clf_i] in prroc_conds:
                for rs in range(random_seed, random_seed +10):
                # for rs in range(random_seed, random_seed +1):
                    if rs == random_seed:
                        clf.fit(train_set,np.array(Ltrain1))
                        exctracted_best_model = clf.fitted_pipeline_.steps[-1][1]
                    
                    # rstrain, rstest1, rstest2, rsLtrain, rsLtest1, rsLtest2 = load_labels(label_key_number, rs, train1_percent, train2_percent, test_percent)
                    # rsdata = split_norm_data(rstrain, rstest1, rstest2)
                    rstrain1, rstrain2, rstest1, rsLtrain1, rsLtrain2, rsLtest1 = load_labels(label_key_number, rs, train1_percent, train2_percent, test_percent)
                    rsdata = split_norm_data(rstrain1, rstest1)
                    rsdata2 = split_norm_data(rstrain2, rstest1)
                    rstrain_set1 = rsdata[data_set_i][1]
                    rstest_set1 = rsdata[data_set_i][2]

                    # rstrain_set2 = rsdata2[data_set_i][1]
                    # rstest_set2 = rsdata2[data_set_i][2]

                    rsmodel = exctracted_best_model.fit(rstrain_set1,np.array(rsLtrain1))
                    y_predict_test1 = rsmodel.predict(rstest_set1)
                    # rstrain_set = rsdata[data_set_i][1]
                    # rstest_set1 = rsdata[data_set_i][2]
                    # rstest_set2 = rsdata[data_set_i][3]
                    # rsmodel = exctracted_best_model.fit(rstrain_set,np.array(rsLtrain))
                    # y_predict_test1 = rsmodel.predict_proba(rstest_set1)
                    # y_predict_test2 = rsmodel.predict_proba(rstest_set2)
                    if write:
                        temp_ypred = []
                        for elt in list(y_predict_test1):
                            # print(elt)
                            temp_ypred.append(float(elt))
                        temp_ypred = y_predict_test1
                        fpr, tpr, thresholds = roc_curve(np.array(rsLtest1), temp_ypred, pos_label=1)
                        precision, recall, thresholds = precision_recall_curve(np.array(rsLtest1), temp_ypred)
                        filename = outdir + 'prroc2/'+str(rs)+ data_set[0]+'-'+cl[clf_i]+'-curves.csv'
                        with open(filename,"w") as f:
                            f.write("\n".join(",".join(map(str, x)) for x in (fpr,tpr,precision, recall)))

def get_ensemble_prroc(label_key_number, random_seed, train1_percent, train2_percent, test_percent,classifiers, cl, outdir, write=True):
    '''calculates the precision recall and receiver operator characteristic curves 

    Inputs:
    prroc_conds (list of str) conatins a list of best classifiers for each dataset 
    label_key_number (int) column number corresponding to which labels to use in the tox_labels.csv file
    random_seed (int) for model instantiation and dataset splitting
    train_percent (float) amount of the dataset used for training, the remaning data will be split into two test sets
    classifiers (list of TPOT classifiers)
    cl (list of str) classifier labels with the same indexing as classifiers
    outdir (str) directory where output files will be saved
    write (boolean) for debugging purposes, False will prevent any files from being saved

    Outputs: returns None, but will save the following files (if the write variable is True)
    random seed dataset - classifier -curves.csv file containing false positive rate, true positive rate, precision and recall for classifier traind on the random seed
    '''
    # train, test1, test2, Ltrain, Ltest1, Ltest2 = load_labels(label_key_number, random_seed, train1_percent, train2_percent, test_percent)
    # data = split_norm_data(train, test1, test2)
    train1, train2, test1, Ltrain1, Ltrain2, Ltest1 = load_labels(label_key_number, random_seed, train1_percent, train2_percent, test_percent)
    data = split_norm_data(train1, test1)

    train_data = pd.read_csv(outdir+'0-level2_train.csv', header=None)
    train_data = train_data.transpose()
    train_data.drop(train_data.tail(1).index,inplace=True)
    Ltrain = get_label_from_l1(outdir, 'train')

    for clf_i in range(len(classifiers)):
        clf = classifiers[clf_i]
        if cl[clf_i] == 'tpotsk':
            clf.fit(train_data,np.array(Ltrain))
            exctracted_best_model = clf.fitted_pipeline_.steps[-1][1]
            for rs in range(10):
                train_data = pd.read_csv(outdir+str(rs)+'-level2_train.csv', header=None)
                train_data = train_data.transpose()
                test_data = pd.read_csv(outdir+str(rs)+'-level2_test.csv', header=None)
                test_data = test_data.transpose()
                train_data.drop(train_data.tail(1).index,inplace=True)
                test_data.drop(test_data.tail(1).index,inplace=True)
                Ltrain = get_label_from_l1(outdir, 'train')
                Ltest = get_label_from_l1(outdir, 'test')
                rsmodel = exctracted_best_model.fit(train_data,np.array(Ltrain)) 
                y_predict_test1 = rsmodel.predict(test_data)

                if write:
                    temp_ypred = []
                    for elt in list(y_predict_test1):
                        # print(elt)
                        temp_ypred.append(float(elt))
                    temp_ypred = y_predict_test1
                    fpr, tpr, thresholds = roc_curve(np.array(Ltest), temp_ypred, pos_label=1)
                    precision, recall, thresholds = precision_recall_curve(np.array(Ltest), temp_ypred)
                    filename = outdir + 'prroc2/'+str(rs)+ 'ensembleall-'+cl[clf_i]+'-curves.csv'
                    with open(filename,"w") as f:
                        f.write("\n".join(",".join(map(str, x)) for x in (fpr,tpr,precision, recall)))

def split_norm_data_dict(train, test1, test2):
    '''loads the data and normalizes by the training set

    Inputs: train, test1, test2 each (list) contains names of the drugs (str) in the data subset 

    Output:
    dictionary where the key is the dataset label (str) and the values are a tuple of pandas dataframes corresponding to train_set,test_set1,test_set2

    '''
    load_train = load_data(train)
    load_test1 = load_data(test1) 
    load_test2 = load_data(test2)
    l = ['sages', 'fp', 'drug_features', 'targetsall']

    out = {}
    for data_i in range(len(load_train)):
        train_set = norm_data_by_train(load_train[data_i],load_train[data_i])
        test_set1 = norm_data_by_train(load_train[data_i],load_test1[data_i])
        test_set2 = norm_data_by_train(load_train[data_i],load_test2[data_i])
        out[l[data_i]]= (train_set,test_set1,test_set2)
    return out

def load_drugs_to_pred_sub(csvfile):
    '''loads the drugs currently in clinical trials which the model will classify along with one of the feature set types

    Inputs:
    csvfile (str) name of the file containing the clinical trial drug information

    Outputs:
    label of drug names (list of str)
    drug features (pandas dataframe)
    '''
    label = []
    data = []
    with open(csvfile) as fo:
        for line in fo:
            split_line = line.replace('\n','').split(',')
            label.append(split_line[0])
            temp = []
            for elt in split_line[1:]:
                temp.append(float(elt))
            data.append(temp)
    return label, pd.DataFrame(data)

def load_drugs_to_pred():
    '''loads all of the clinical trial drugs and their data sets 

    Inputs: None

    Outputs:dictionary where the key is the feature type (str) and the value is the drug data matrix data (pandas dataframe)
    '''
    out = {}
    file_types = ['sages','fp','targetsall','drug_features']
    for name in file_types:
        label, data = load_drugs_to_pred_sub('trials_'+name+'.csv')
        out[name] = data
    return out,label

def load_level2_drugspred(outdir,rs):
    '''loads the outputs from the individual classifiers to use as training for the ensemble

    Inputs: 
    outdir (str) directory of the files to load
    rs (int) random seed for determining which subset of the data to use

    Outputs: train data (pandas dataframe) of 
    '''
    train_data = pd.read_csv(outdir+str(rs)+'predtrialdrugs-level2.csv', header=None)
    train_data = train_data.transpose()
    train_data.drop(train_data.head(1).index,inplace=True)
    return train_data

def pred_trials_level1(pred_conds, label_key_number, train1_percent, train2_percent, test_percent,classifiers, cl, outdir, write=True):
    '''individual classifier (trained on the test set) predictions of drugs in clinical trials

    Inputs:
    pred_conds (list of str) classifier types that performed the best for each feature type set
    label_key_number (int) column number corresponding to which labels to use in the tox_labels.csv file
    train_percent (float) amount of the dataset used for training, the remaning data will be split into two test sets
    classifiers (list of TPOT classifiers)
    cl (list of str) classifier labels with the same indexing as classifiers
    outdir (str) directory where output files will be saved
    write (boolean) for debugging purposes, False will prevent any files from being saved
    
    Outputs: returns None, but will save the following files (if the write variable is True)
    random seed predtrialdrugs-level2.csv - saves the predictions of the individual classifiers for use in the ensemble predictor
    '''
    pred_drugs_data_dict, pred_drug_names = load_drugs_to_pred()
    for clf_i in range(len(classifiers)):
        for data_info in pred_conds:
            data_name = data_info.split('-')[0]
            if data_name+ '-' + cl[clf_i] in pred_conds:
                print('****************')
                print(data_name)
                train, test1, test2, Ltrain, Ltest1, Ltest2 = load_labels(label_key_number, 0, train1_percent, train2_percent, test_percent)
                train_set,test_set1,test_set2 = split_norm_data_dict(train, test1, test2)[data_name]
                clf = classifiers[clf_i]
                clf.fit(train_set,np.array(Ltrain))
                exctracted_best_model = clf.fitted_pipeline_.steps[-1][1]
                for rs in range(10):
                    train, test1, test2, Ltrain, Ltest1, Ltest2 = load_labels(label_key_number, rs, train1_percent, train2_percent, test_percent)
                    train_set,test_set1,test_set2 = split_norm_data_dict(train, test1, test2)[data_name]
                    rsmodel = exctracted_best_model.fit(train_set,np.array(Ltrain))
                    try:
                        y_predict = rsmodel.predict_proba(pred_drugs_data_dict[data_name])
                        if write:
                            fout4= open(outdir+str(rs)+'predtrialdrugs-level2.csv','a+')
                            fout4.write(str(data_name))
                            for elt in list(y_predict):
                                fout4.write(',' + str(elt[0]))
                            fout4.write('\n')
                            fout4.close()
                    except:
                        y_predict = rsmodel.predict(pred_drugs_data_dict[data_name])
                        if write:
                            fout4= open(outdir+str(rs)+'predtrialdrugs-level2.csv','a+')
                            fout4.write(str(data_name))
                            for elt in list(y_predict):
                                fout4.write(','+str(elt))
                            fout4.write('\n')
                            fout4.close()

def pred_trials_level2(outdir,write=True):
    '''ensemble predictor trained on test set for use predicting drugs in clinical trials
    
    Inputs: 
    outdir (str) directory of where to save the written files 
    write (boolean) for debugging purposes, False will prevent any files from being saved

    Outputs:returns None, but will save the following files (if the write variable is True)
    final_predictions_predtrialdrugs.csv - each column represents 
    '''
    pred_drugs_data_dict, pred_drug_names = load_drugs_to_pred()
    if write:
        fout4= open(outdir+'final_predictions_predtrialdrugs.csv','a+')
        fout4.write(','.join(pred_drug_names) +'\n')
        fout4.close()
    for rs in range(10):
        for clf_i in range(len(classifiers)):
            if cl[clf_i]=='tpotsk':
                if rs == 0:
                    train_data = pd.read_csv(outdir+'0-level2_train.csv', header=None)
                    train_data = train_data.transpose()
                    train_data.drop(train_data.tail(1).index,inplace=True)
                    Ltrain = get_label_from_l1(outdir, 'train')
                    clf = classifiers[clf_i]
                    clf.fit(train_data,np.array(Ltrain))
                    exctracted_best_model = clf.fitted_pipeline_.steps[-1][1]
                    y_predict = exctracted_best_model.predict(train_data)
                
                train_data = pd.read_csv(outdir+str(0)+'-level2_train.csv', header=None)
                train_data = train_data.transpose()
                train_data.drop(train_data.tail(1).index,inplace=True)
                rsmodel = exctracted_best_model.fit(train_data,np.array(Ltrain))
                level2_drugstopred_data = load_level2_drugspred(outdir,rs)
                y_predict_test1 = rsmodel.predict(level2_drugstopred_data)
                if write:
                    fout4= open(outdir+'final_predictions_predtrialdrugs.csv','a+')
                    for elt in list(y_predict_test1):
                        fout4.write(str(elt)+ ',')
                    fout4.write('\n')
                    fout4.close()

# def get_prroc_averages(prroc_conds,classifiers, cl, outdir, write=True):
#     ''' averages the classifier performance for all cross validation steps

#     Inputs:
#     prroc_conds (list of str) the results from which classifier and feature set combo to average
#     outdir (str) directory of where to save the written files 
#     classifiers (list of TPOT classifiers)
#     cl (list of str) classifier labels with the same indexing as classifiers
#     write (boolean) for debugging purposes, False will prevent any files from being saved

#     Outputs:returns None, but will save the following files (if the write variable is True)
#     newprroc/level2_summary.csv average performance metrics for the hyperparameter tuned enesemble classifiers
#     '''
#     for clf_i in range(len(classifiers)):
#         if 'all-'+cl[clf_i] in prroc_conds:
#             for rs in range(10):
#                 train_data = pd.read_csv(outdir+str(rs)+'-level2_train.csv', header=None)
#                 train_data = train_data.transpose()
#                 train_data.drop(train_data.tail(1).index,inplace=True)
#                 Ltrain = get_label_from_l1(outdir, 'train')
#                 test_data = pd.read_csv(outdir+str(rs)+'-level2_test.csv', header=None)
#                 average_predictor = test_data.mean().to_list()[:-1]
#                 test_data = test_data.transpose()
#                 test_data.drop(test_data.tail(1).index,inplace=True)
#                 Ltest = get_label_from_l1(outdir, 'test')
                
#                 if rs == 0:
#                     clf = classifiers[clf_i]
#                     clf.fit(train_data,np.array(Ltrain))
#                     rsmodel = clf.fitted_pipeline_.steps[-1][1]

#                 y_predict_test1 = rsmodel.predict_proba(test_data)
#                 temp_average_predictor = []
#                 for elt in average_predictor:
#                     if elt <=0.5:
#                         temp_average_predictor.append(1)
#                     else:
#                         temp_average_predictor.append(0)
#                 acc,aroc,f1_val,precision_val,recall_val,mcc = evaluate(np.array(Ltest), temp_average_predictor)
#                 if write:
#                     out_str = str(rs)+','+str(acc)+','+str(aroc)+','+str(f1_val)+','+str(precision_val)+','+str(recall_val) +','+str(mcc) +',allaverage\n'
#                     fout0 = open(outdir+'newprroc/level2_summary.csv', '+a')
#                     fout0.write(out_str)
#                     fout0.close()

#                     temp_ypred = []
#                     for elt in list(y_predict_test1):
#                         temp_ypred.append(float(elt[1]))
#                     fpr, tpr, thresholds = roc_curve(np.array(Ltest), temp_ypred, pos_label=1)
#                     precision, recall, thresholds = precision_recall_curve(np.array(Ltest), temp_ypred)

#                     filename = outdir + 'newprroc/'+str(rs)+'all-'+cl[clf_i]+'-curves.csv'
#                     with open(filename,"w") as f:
#                         f.write("\n".join(",".join(map(str, x)) for x in (fpr,tpr,precision, recall)))
                    
#                     fpr, tpr, thresholds = roc_curve(np.array(Ltest), average_predictor, pos_label=1)
#                     precision, recall, thresholds = precision_recall_curve(np.array(Ltest), average_predictor)

#                     filename = outdir +'newprroc/'+ str(rs)+'all-average-curves.csv'
#                     with open(filename,"w") as f:
#                         f.write("\n".join(",".join(map(str, x)) for x in (fpr,tpr,precision, recall)))

# def get_average_performance_l1(outdir, infile,outfile, write=True):
#     '''averages the classifier performance for all cross validation steps for the classifiers trained on each feature set

#     Inputs:
#     outdir (str) directory of where to save the written files 
#     infile (str) the file that contains all the performance metrics for the model
#     outfile (str) name of the file to save the average performance to

#     Output:returns None, but will save the following file  - outfile
#     '''
#     out = {}
#     line_i = 0
#     with open(outdir+infile) as fo:
#         for line in fo:
#             if line_i != 0:
#                 split_line = line[:-1].split(',')
#                 current_key = split_line[1] + '_' + split_line[8]
#                 if current_key not in out:
#                     out[current_key] = {'Accuracy':[], 'AUROC':[],'F1':[],'Precision':[],'Recall':[], 'MCC':[]}
#                 out[current_key]['Accuracy'] = out[current_key]['Accuracy'] + [float(split_line[2])]
#                 out[current_key]['AUROC'] = out[current_key]['AUROC'] + [float(split_line[3])]
#                 out[current_key]['F1'] = out[current_key]['F1'] + [float(split_line[4])]
#                 out[current_key]['Precision'] = out[current_key]['Precision'] + [float(split_line[5])]
#                 out[current_key]['Recall'] = out[current_key]['Recall'] + [float(split_line[6])]
#                 out[current_key]['MCC'] = out[current_key]['MCC'] + [float(split_line[7])]
#             line_i += 1
#     l = ['Accuracy', 'AUROC','F1','Precision','Recall','MCC']
#     if write:
#         out_header = 'Model'
#         for k in l:
#             out_header = out_header + ',' + k
#         fout = open(outdir+outfile, 'a+', encoding="utf-8-sig")
#         fout.write(out_header + '\n')
#         fout.close()
#     for dict_key in out.keys():
#         line_out = dict_key
#         for k in l:
#             line_out = line_out + ',' + str(round(sum(out[dict_key][k])/10,3)) + "±" + str(round(statistics.stdev(out[dict_key][k]),3))
#             #line_out = line_out + ',' + str(out[dict_key][k]/10)
#         if write:
#             fout = open(outdir+outfile, 'a+', encoding="utf-8-sig")
#             fout.write(line_out + '\n')
#             fout.close()

def make_level2_data(outdir,test_or_train, best_models):
    '''reformats the predictions from the classifiers trained on a subset of features to be used as training and testing data for the ensemble

    Inputs:
    outdir (str) directory of where to save the written files 
    test_or_train (str) 'test' or 'train' label indicating which dataset to transpose
    best_models (str) represents the label of the output of the classifier trained on a certain feature subset to transpose

    Outputs:returns None, but will save the following file
    random seed -level2_ test_or_train .csv - the data for use in training the ensemble
    '''
    for rs in range(10):
        out = ''
        for bm in best_models:
            with open(outdir+ bm+ '-' + str(rs)+ '-level2_'+test_or_train+'.csv') as fo:
                for line in fo:
                    out = out + line
        fout = open(outdir+ str(rs)+'-level2_'+test_or_train+'.csv','a+')
        fout.write(out)
        fout.close()

# def get_average_performance_l2(outdir, infile,outfile):
#     '''averages the ensemble performance for all cross validation steps

#     Inputs:
#     outdir (str) directory of where to save the written files 
#     infile (str) the file that contains all the performance metrics for the model
#     outfile (str) name of the file to save the average performance to

#     Output:returns None, but will save the following file  - outfile
#     '''
#     out = {}
#     with open(outdir+infile) as fo:
#         line_i = 0
#         for line in fo:
#             if line_i != 0:
#                 split_line = line.replace('\n','').split(',')
#                 current_key = split_line[7]
#                 if current_key not in out:
#                     out[current_key] = {'Accuracy':0, 'AUROC':0,'F1':0,'Precision':0,'Recall':0, 'MCC':0}
#                 out[current_key]['Accuracy'] = out[current_key]['Accuracy'] + float(split_line[1])
#                 out[current_key]['AUROC'] = out[current_key]['AUROC'] + float(split_line[2])
#                 out[current_key]['F1'] = out[current_key]['F1'] + float(split_line[3])
#                 out[current_key]['Precision'] = out[current_key]['Precision'] + float(split_line[4])
#                 out[current_key]['Recall'] = out[current_key]['Recall'] + float(split_line[5])
#                 out[current_key]['MCC'] = out[current_key]['MCC'] + float(split_line[6])
#             line_i+=1
#     l = ['Accuracy', 'AUROC','F1','Precision','Recall','MCC']
#     for dict_key in out.keys():
#         line_out = dict_key
#         for k in l:
#             line_out = line_out + ',' + str(out[dict_key][k]/10)
#         fout = open(outdir+outfile, 'a+')
#         fout.write(line_out + '\n')
#         fout.close()

# Variable Values
parametersRF = {'criterion': ['entropy', 'gini'],'max_depth': list(np.linspace(10, 500, 10, dtype = int)) + [None],'max_features': ['auto', 'sqrt','log2', None],'min_samples_leaf': [2, 15],'min_samples_split': [5, 15],'n_estimators': list(np.linspace(150, 500, 10, dtype = int))}
parametersMLP = {'activation': ['identity', 'loginst', 'tanh','relu'],'hidden_layer_sizes': list(np.linspace(25,400, 10, dtype = int)),'solver': ['lbfgs', 'sgd','adam']}
parametersXGB = {'max_depth': list(np.linspace(10, 500, 10, dtype = int)) + [None],'n_estimators': list(np.linspace(150, 500, 10, dtype = int))}
my_search = TPOTClassifier( population_size= 24, offspring_size= 12, verbosity= 2, early_stop= 12, scoring = 'accuracy', cv = 5, generations= 5,random_state=0,
                        config_dict={'sklearn.ensemble.RandomForestClassifier': parametersRF,
                            'sklearn.neural_network.MLPClassifier': parametersMLP,
                            'xgboost.XGBClassifier':parametersXGB
                                })
og_search = TPOTClassifier(generations= 5, population_size= 24, offspring_size= 12, verbosity= 2, early_stop= 12, config_dict='TPOT NN', cv = 5, scoring = 'accuracy', random_state=0,)
classifiers = [my_search, og_search]
cl = ['tpotsk','tpotdefault']

outdir = 'ensemble_train_603010/'
#bm = ['sages-tpotsk', 'fp-tpotsk', 'drug_features-tpotdefault', 'targetsall-tpotsk']
# get_prroc(bm, 1, 0, 0.6, 0.3, 0.1,classifiers, cl, outdir, write=True)
get_ensemble_prroc(1, 0, 0.6, 0.3, 0.1,classifiers, cl, outdir, write=True)


#outdir = 'retrained2/'
# tuning_level1(1, 0, 10, 0.9 ,classifiers, cl, outdir)
# make_level2_data(outdir,'train', ['sages-tpotdefault', 'fp-tpotdefault', 'drug_features-tpotsk', 'targetsall-tpotdefault' ])
# make_level2_data(outdir,'test', ['sages-tpotdefault', 'fp-tpotdefault', 'drug_features-tpotsk', 'targetsall-tpotdefault' ])
#tuning_level2(classifiers, cl, outdir, write = True)
#

#outdir = 'ensemble_train_9010/'
#tuning_level1(1, 0, 10, 0.9 ,classifiers, cl, outdir)
# bm = ['sages-tpotsk', 'fp-tpotsk', 'drug_features-tpotdefault', 'targetsall-tpotdefault']
# make_level2_data(outdir,'train',bm )
# make_level2_data(outdir,'test', bm)
# tuning_level2(classifiers, cl, outdir, write = True)


# # outdir = 'ensemble_train_702010/'
# outdir = 'normtargetensemble_train_702010/'
# # tuning_level1(1, 0, 10, 0.7, 0.2, 0.1,classifiers, cl, outdir)
# # bm = ['sages-tpotsk', 'fp-tpotsk', 'drug_features-tpotdefault', 'targetsall-tpotdefault']
# bm = ['sages-tpotsk', 'fp-tpotsk', 'drug_features-tpotdefault', 'targetsall-tpotsk']
# make_level2_data(outdir,'train',bm )
# make_level2_data(outdir,'test', bm)
# tuning_level2(classifiers, cl, outdir, write = True)

# # outdir = 'ensemble_train_801010/'
# outdir = 'normtargetensemble_train_801010/'
# # tuning_level1(1, 0, 10, 0.8, 0.1, 0.1,classifiers, cl, outdir)
# # bm = ['sages-tpotsk', 'fp-tpotsk', 'drug_features-tpotsk', 'targetsall-tpotdefault']
# bm = ['sages-tpotsk', 'fp-tpotsk', 'drug_features-tpotsk', 'targetsall-tpotdefault']
# make_level2_data(outdir,'train',bm )
# make_level2_data(outdir,'test', bm)
# tuning_level2(classifiers, cl, outdir, write = True)

# outdir = 'ensemble_train_603010/'
# # tuning_level1(1, 0, 10, 0.6, 0.3, 0.1,classifiers, cl, outdir)
# bm = ['sages-tpotsk', 'fp-tpotsk', 'drug_features-tpotdefault', 'targetsall-tpotsk']
# make_level2_data(outdir,'train',bm )
# make_level2_data(outdir,'test', bm)
# tuning_level2(classifiers, cl, outdir, write = True)
