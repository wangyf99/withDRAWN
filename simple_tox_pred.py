import random
import math
import numpy as np
import pandas as pd
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

def load_labels(label_key_number, random_seed, train_percent, test_percent):
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
    rearrange_positive = random.sample(positive, len(positive))
    rearrange_negative = random.sample(negative, len(negative))

    number_pos1 = math.floor(len(positive)*(train_percent))
    number_neg1 = math.floor(len(negative)*(train_percent))

    number_postest = math.floor(len(positive)*(test_percent))
    number_negtest = math.floor(len(negative)*(test_percent))

    train1 = rearrange_positive[:number_pos1] + rearrange_negative[:number_neg1] 
    test1 = rearrange_positive[number_pos1:number_pos1+ number_postest] + rearrange_negative[number_neg1:number_neg1+number_negtest]
    test2 = rearrange_positive[number_pos1+ number_postest:] + rearrange_negative[number_neg1+number_negtest:]
    

    Ltrain1 = [1]*len(rearrange_positive[:number_pos1]) + [0]*len(rearrange_negative[:number_neg1])
    Ltest1 = [1]*len(rearrange_positive[number_pos1:number_pos1+number_postest]) + [0]*len(rearrange_negative[number_neg1:number_neg1+number_negtest])
    Ltest2 = [1]*len(rearrange_positive[number_pos1+number_postest:]) + [0]*len(rearrange_negative[number_neg1+number_negtest:])
    return train1, test1, test2, Ltrain1, Ltest1, Ltest2

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
    all_data = []
    sages_out = load_nongraph(drug_names_list, 'sages.csv')
    fp_out = load_nongraph(drug_names_list, 'fp.csv')
    drug_features_out = load_nongraph(drug_names_list, 'drug_features.csv')
    targetsall = load_nongraph(drug_names_list, 'targetsall.csv')
    for row_index in range(len(sages_out)):
        all_data.append(sages_out[row_index] + fp_out[row_index] + drug_features_out[row_index] + targetsall[row_index])
    return pd.DataFrame(all_data)

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

def split_norm_data(train, test1):
    ''' loads the data and normalizes by the training set

    Inputs: train, test1, test2 each (list) contains names of the drugs (str) in the data subset 

    Output:
    list of tuples where the first index of the tuple is the dataset label (str) and the remainder are pandas dataframes corresponding to train_set,test_set1,test_set2
    '''
    load_train = load_data(train)
    load_test1 = load_data(test1) 
    train_set = norm_data_by_train(load_train,load_train)
    test_set1 = norm_data_by_train(load_train,load_test1)
    return train_set,test_set1

def eval_and_write(rs,ts,rsLtest1,y_predict_test1,y_predictproba_test1, outdir, out_class_code):
    acc,aroc,f1_val,precision_val,recall_val,mcc = evaluate(np.array(rsLtest1), y_predict_test1)
    out_str = str(rs)+','+str(ts)+',alldataonelevel,'+str(acc)+','+str(aroc)+','+str(f1_val)+','+str(precision_val)+','+str(recall_val) +','+str(mcc)+','+ out_class_code +'\n'
    fout0 = open(outdir+'simple_level0_summary.csv', '+a')
    fout0.write(out_str)
    fout0.close()

    temp_ypred = []
    for elt in list(y_predictproba_test1):
        temp_ypred.append(float(elt[1]))
    fpr, tpr, thresholds = roc_curve(np.array(rsLtest1), temp_ypred, pos_label=1)
    precision, recall, thresholds = precision_recall_curve(np.array(rsLtest1), temp_ypred)
    filename = outdir + 'prroc/'+str(ts)+'_'+str(rs)+'nonensembleall-'+out_class_code+'-curves.csv'
    with open(filename,"w") as f:
        f.write("\n".join(",".join(map(str, x)) for x in (fpr,tpr,precision,recall)))

def update_miss_dict(miss_dict, rstest1,rsLtest1,y_predict_test1):
    for predict_index in range(len(rstest1)):
        # print(rsLtest1[predict_index])
        # print(y_predict_test1[predict_index])
        # print(rsLtest1)
        # print(y_predict_test1)
        if int(rsLtest1[predict_index]) != int(y_predict_test1[predict_index]):
            # print('here')
            if rstest1[predict_index] not in miss_dict:
                miss_dict[rstest1[predict_index]] = 1
            else:
                miss_dict[rstest1[predict_index]] = miss_dict[rstest1[predict_index]] + 1
        # print('~~~~~~~~~~~~~~~~~~~')
    return miss_dict

def tuning_level1(label_key_number, random_seed, train_percent, test_percent, classifiers, cl, outdir, write=False):
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
    miss_dict_train = {}
    miss_dict_test = {}
    train1, test1, test2, Ltrain1, Ltest1, Ltest2 = load_labels(label_key_number, random_seed, train_percent, test_percent)
    train_set,test_set1 = split_norm_data(train1, test1)
    if write:
        fout0 = open(outdir+'simple_level0_summary.csv', '+a')
        fout0.write('RandomSeed,TestSet,Data,Accuracy,AUROC,F1,Precision,Recall,MCC,Classifier\n')
        fout0.close()
        
    for clf_i in range(len(classifiers)):
        clf = classifiers[clf_i]
        clf.fit(train_set,np.array(Ltrain1))
        exctracted_best_model = clf.fitted_pipeline_.steps[-1][1]
        out_class_code = cl[clf_i]
        if write:
            clf.export(outdir+out_class_code+ '-onelevel-tpot_exported_pipeline.py')
        for rs in range(random_seed, random_seed +10):
            rstrain1, rstest1, rstest2, rsLtrain1, rsLtest1, rsLtest2 = load_labels(label_key_number, rs, train_percent, test_percent)
            rstrain_set1, rstest_set1 = split_norm_data(rstrain1, rstest1)
            temp, rstest_set2 = split_norm_data(rstrain1, rstest2)
            
            rsmodel = exctracted_best_model.fit(rstrain_set1,np.array(rsLtrain1))

            y_predict_test1 = rsmodel.predict(rstest_set1)
            y_predictproba_test1 = rsmodel.predict_proba(rstest_set1)
            y_predict_test2 = rsmodel.predict(rstest_set2)
            y_predictproba_test2 = rsmodel.predict_proba(rstest_set2)

            miss_dict_train = update_miss_dict(miss_dict_train, rstrain1,rsLtrain1,rsmodel.predict(rstrain_set1))
            miss_dict_test = update_miss_dict(miss_dict_test, rstest1,rsLtest1,y_predict_test1)
            miss_dict_test = update_miss_dict(miss_dict_test, rstest2,rsLtest2,y_predict_test2)

            if write:
                eval_and_write(rs,1,rsLtest1,y_predict_test1,y_predictproba_test1, outdir, out_class_code)
                eval_and_write(rs,2,rsLtest2,y_predict_test2,y_predictproba_test2, outdir, out_class_code)
    if write:
        for elt in miss_dict_train:
            out_str = elt + ',' + str(miss_dict_train[elt])+'\n'
            fout0 = open(outdir+'missclassified_drugs_train.csv', '+a')
            fout0.write(out_str)
            fout0.close()
        for elt in miss_dict_test:
            out_str = elt + ',' + str(miss_dict_test[elt])+'\n'
            fout0 = open(outdir+'missclassified_drugs_train.csv', '+a')
            fout0.write(out_str)
            fout0.close()








def get_prroc(label_key_number, random_seed, train_percent,test_percent, classifiers, cl, outdir,best_classifier, write=True):
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
    #train, test1, test2, Ltrain, Ltest1, Ltest2 
    train, test, Ltrain, Ltest = load_labels(label_key_number, random_seed, train_percent, test_percent)
    train_set,test_set1 = split_norm_data(train, test)

    for clf_i in range(len(classifiers)):
        clf = classifiers[clf_i]
        if cl[clf_i] == best_classifier:
            clf.fit(train_set,np.array(Ltrain))
            exctracted_best_model = clf.fitted_pipeline_.steps[-1][1]
            for rs in range(random_seed, random_seed +10):
                rstrain1, rstest1, rsLtrain1, rsLtest1 = load_labels(label_key_number, rs, train_percent, test_percent)
                rstrain_set1, rstest_set1 = split_norm_data(rstrain1, rstest1)
                
                rsmodel = exctracted_best_model.fit(rstrain_set1,np.array(rsLtrain1))
                y_predict_test1 = rsmodel.predict_proba(rstest_set1)

                if write:
                    temp_ypred = []
                    for elt in list(y_predict_test1):
                        # print(elt)
                        temp_ypred.append(float(elt[1]))
                    fpr, tpr, thresholds = roc_curve(np.array(Ltest), temp_ypred, pos_label=1)
                    precision, recall, thresholds = precision_recall_curve(np.array(Ltest), temp_ypred)
                    filename = outdir + 'prroc/'+str(rs)+ 'nonensembleall-'+cl[clf_i]+'-curves.csv'
                    with open(filename,"w") as f:
                        f.write("\n".join(",".join(map(str, x)) for x in (fpr,tpr,precision,recall)))



# def get_missclassified(label_key_number, random_seed, train_percent,test_percent, classifiers, cl, outdir,best_classifier, write=True):
#     '''
#     '''
#     miss_dict = {}
#     train, test, Ltrain, Ltest = load_labels(label_key_number, random_seed, train_percent, test_percent)
#     train_set,test_set1 = split_norm_data(train, test)
#     if write:
#         fout0 = open(outdir+'simple_level0_summary.csv', '+a')
#         fout0.write('RandomSeed,Data,Accuracy,AUROC,F1,Precision,Recall,MCC,Classifier\n')
#         fout0.close()
        
#     for clf_i in range(len(classifiers)):
#         clf = classifiers[clf_i]
#         if cl[clf_i] == best_classifier:
#             clf.fit(train_set,np.array(Ltrain))
#             exctracted_best_model = clf.fitted_pipeline_.steps[-1][1]
#             # for rs in range(random_seed, random_seed +10):
#             rstrain1, rstest1, rsLtrain1, rsLtest1 = load_labels(label_key_number, random_seed, train_percent, test_percent)
#             rstrain_set1, rstest_set1 = split_norm_data(rstrain1, rstest1)
            
#             rsmodel = exctracted_best_model.fit(rstrain_set1,np.array(rsLtrain1))

#             y_predict_test1 = rsmodel.predict(rstest_set1)
#             y_predict_train = rsmodel.predict(rstrain_set1)
            
#             miss_dict = update_miss_dict(miss_dict,rstest1,rsLtest1,y_predict_test1)
#             miss_dict = update_miss_dict(miss_dict,rstrain1,rsLtrain1,y_predict_train)

#     if write:
#         for elt in miss_dict:
#             out_str = elt + ',' + str(miss_dict[elt])+'\n'
#             fout0 = open(outdir+'missclassified_drugs.csv', '+a')
#             fout0.write(out_str)
#             fout0.close()

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

# cl = ['tpotdefault']

# get_prroc(1, 0, 0.9,classifiers, cl, 'ensemble_train_603010/', write=True)

tuning_level1(1, 0, 0.8,0.1,classifiers, cl, 'simple_8010/', write=True)
# best_classifier ='tpotsk'
# get_missclassified(1, 0, 0.8,0.1, classifiers, cl, 'simple_8010/',best_classifier, write=True)
# get_prroc(1, 0, 0.8,0.1,classifiers, cl, 'simple_8010/',best_classifier, write=True)