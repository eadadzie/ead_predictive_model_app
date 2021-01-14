# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 13:59:59 2021

@author: eadob
"""

### Importing Relevant Packages
import pandas as pd
import numpy as np
import streamlit as st
import base64
import io

from sklearn import model_selection
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score, mean_absolute_percentage_error
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

from ead_visualization_st import plot_bar

###########################################################################################
##### Encoding the class
def encode_data(dataframe, col_name='class'):
	unique_list = []
	for x in dataframe[col_name]:
		if x not in unique_list:
			unique_list.append(x)			
	encode = []
	for i in dataframe[col_name]:
		if i in unique_list:
			encode.append(unique_list.index(i))	
	dataframe[col_name] = [int(z) for z in encode]
	return dataframe


# Model building
def filedownload(df, f_ext='csv', algm_name='Algorithm'):
    # csv = df.to_csv(index=False)
    # b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    if f_ext == 'csv':
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="model_performance.csv">Download CSV File</a>'
    elif f_ext == 'txt': 
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="model_report-{algm_name}.txt">Download TXT File - {algm_name}</a>' 
    #elif f_ext == 'png':
    return href


# # Parameters
n_trees = 100
l_rate = 0.1
m_depth = None
## Build ML models
def model_development(dataframe, y_val, x_vals, split_size, rand_state, scoring, algm_models, sel_algms=['KNN','CART','NB'], out_put='classification'):  # sel_algms=['KNN','CART','NB']
	y_col = y_val
	x_cols = x_vals  # list(dataframe.columns)
	#x_cols.remove(y_col)
    
	X = dataframe.loc[:, x_cols]
	Y = dataframe.loc[:, y_col]     
    
	validation_size = (100 - split_size) / 100
	seed = rand_state
    
	# global X_train, X_validation, Y_train, Y_validation, models
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
	
    # evaluate each model in turn
	results = []
	names = []
	cv_means = []
	cv_std = []
    # Looping to create models for Cross Validation
	for name, model in algm_models.items():
		kfold = model_selection.KFold(n_splits=10, random_state=None)
		cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
		results.append(cv_results)
		names.append(name)
		cv_means.append(cv_results.mean())
		cv_std.append(cv_results.std())
    
	# Printing prediction results
	dict_report = {}
	dict_cm = {}
	predict_list = []
	for name in sel_algms:
		prediction = algm_models[name].fit(X_train, Y_train).predict(X_validation)
		if out_put.lower() == 'classification':
			if scoring == 'accuracy':
				predict_list.append(accuracy_score(Y_validation, prediction))
			elif scoring == 'f1':
				predict_list.append(f1_score(Y_validation, prediction))
			elif scoring == 'f1_weighted':
				predict_list.append(f1_score(Y_validation, prediction, average='weighted'))
			elif scoring == 'roc_auc':
				predict_list.append(roc_auc_score(Y_validation, prediction))
			elif scoring == 'recall':
				predict_list.append(recall_score(Y_validation, prediction))
			elif scoring == 'precision':
				predict_list.append(precision_score(Y_validation, prediction))
            ### Confusion Matrices
			dict_cm[name] = confusion_matrix(Y_validation, prediction)  #, columns=list(Y.unique()), index=list(Y.unique()))
            ### Classification report
			dict_report[name] = classification_report(Y_validation, prediction)
		elif out_put.lower() == 'regression':
			if scoring == 'r2':
				predict_list.append(r2_score(Y_validation, prediction))
			elif scoring == 'neg_mean_squared_error':
				predict_list.append(mean_squared_error(Y_validation, prediction))
			elif scoring == 'neg_mean_absolute_error':
				predict_list.append(mean_absolute_error(Y_validation, prediction))
			elif scoring == 'neg_root_mean_squared_error':
				predict_list.append(mean_squared_error(Y_validation, prediction, squared=False))
			elif scoring == 'neg_mean_absolute_percentage_error':
				predict_list.append(mean_absolute_percentage_error(Y_validation, prediction))
			elif scoring == 'explained_variance':
				predict_list.append(explained_variance_score(Y_validation, prediction))  
    ## Show summary results
	st.markdown('#\n')
	df_results = pd.DataFrame({'Algorithms':algm_models.keys(), 'CV_means':np.array(cv_means).T, 'CV_std':np.array(cv_std).T, 'Prediction':np.array(predict_list).T})
	st.markdown('\n')
	st.success(f'**Summary Results** based on Model {scoring}')
	st.table(df_results)
    
    # Download Results
	st.info('Download **Summary Results**')
	st.markdown(filedownload(df_results, f_ext='csv'), unsafe_allow_html=True)
	st.markdown('#\n')
    
    ### Outputs relating to classification
	if out_put.lower() == 'regression':
        ### Ploting results graph
		st.success('Graph of the **Summary Results**')
		models_results = [df_results[coln].values.tolist() for coln in df_results.columns]
		fig_bar, ax_bar = plt.subplots()
		plot_bar(models_results, x_lab="Algorithms", y_lab="Accuracy (%)")
		st.pyplot(fig_bar)
		st.markdown('#\n')
        
        ### Generating and downloading Regression reports
		st.success('**Regression Report** of Model(s)')
		dict_reg_rpt = {}
		for name in sel_algms:
			prediction = algm_models[name].fit(X_train, Y_train).predict(X_validation)
			dict_reg_rpt[name] = {      'r2' : round(r2_score(Y_validation, prediction), 4),
                    'neg_mean_squared_error' : round(mean_squared_error(Y_validation, prediction), 4),
                   'neg_mean_absolute_error' : round(mean_absolute_error(Y_validation, prediction), 4),
               'neg_root_mean_squared_error' : round(mean_squared_error(Y_validation, prediction, squared=False), 4),
        'neg_mean_absolute_percentage_error' : round(mean_absolute_percentage_error(Y_validation, prediction), 4),
                        'explained_variance' : round(explained_variance_score(Y_validation, prediction), 4)
                                 }
		if len(sel_algms) == 1:
			num_cols = 1
		elif len(sel_algms) in [2, 3, 4]:
			num_cols = 2
		elif len(sel_algms) >= 4:
			num_cols = 3  
		else: num_cols = 3
        
		cnt7 = 0
		reg_rpt_cols = st.beta_columns(num_cols)
		st.info('Download **Reports**')
		dl_reg_cols = st.beta_columns(num_cols)
		for name, report in dict_reg_rpt.items():
            # Display reports in report columns/grid
			text_rpt = ''
			for k, v in dict_reg_rpt[name].items():
				text_rpt += f'{k} : {v}\n'
			reg_rpt_cols[cnt7].header(name)
			reg_rpt_cols[cnt7].text(text_rpt)
            # Display download links download columns/grid               
			data = pd.read_csv(io.StringIO(text_rpt))
			dl_reg_cols[cnt7].markdown(filedownload(data, f_ext='txt', algm_name=name), unsafe_allow_html=True)
			cnt7 += 1
			if cnt7 > num_cols - 1:
				cnt7 = 0
		st.markdown('#\n')

    ### Outputs relating to classification
	if out_put.lower() == 'classification':
        ## Show confusion matrices
		st.success('**Confusion Matrix** of Model(s)')
		if len(sel_algms) == 1:
			num_cols = 1
		elif len(sel_algms) in [2, 3, 4]:
			num_cols = 2
		elif len(sel_algms) >= 4:
			num_cols = 3  
		else: num_cols = 3
        # Making graphs arrange in rows and maximum of 3 columns base on the number of models    
		cnt = 0
		plot_cols = st.beta_columns(num_cols)
		st.info('Right click on image to save **Confusion Matrix** as png file')
		#cm_cols = st.beta_columns(num_cols)
		for name, model in dict_cm.items():
			fig, ax = plot_confusion_matrix(conf_mat=model, show_absolute=True, show_normed=True, colorbar=True)
			ax.set_title(name)
			plot_cols[cnt].pyplot(fig)
			#cm_cols[cnt].markdown(f'Download File {name}.png')
			cnt += 1
			if cnt > num_cols - 1:
				cnt = 0
		st.markdown('#\n')
        #st.pyplot()

        ## Classification Report
		st.success('**Classification Report** of Model(s)')
		cnt2 = 0
		report_cols = st.beta_columns(num_cols)
		st.info('Download **Reports**')
		dl_rpt_cols = st.beta_columns(num_cols)
		for name, report in dict_report.items():
            # Display reports in report columns/grid
			report_cols[cnt2].header(name)
			report_cols[cnt2].text(dict_report[name]) 
            # Display download links download columns/grid               
			data = pd.read_csv(io.StringIO(report))
			dl_rpt_cols[cnt2].markdown(filedownload(data, f_ext='txt', algm_name=name), unsafe_allow_html=True)
			cnt2 += 1
			if cnt2 > num_cols - 1:
				cnt2 = 0
		st.markdown('#\n')                
    ## Return the models for boundary ploting
	#return models






