
############################ FUNCTIONS #############################
import pandas as pd


##### Scale data
def scale_data(dataframe, ex_list= [], transform= 'norm'):
	array_norm = dataframe.values
	array_stand = dataframe.values
	# Normalize the dataset except columns in exempt list
	for i in range(len(array_norm[1])):
		if (i not in ex_list):
			value_min = min(array_norm [:,i]) 
			value_max = max(array_norm [:,i]) 
			min_max = value_max - value_min
			array_norm [:,i]= (array_norm [:,i] - value_min) / min_max
	# Standardize the dataset except Class Label
	for i in range(len(array_stand[1])): 
		if (i not in ex_list):
			array_stand [:,i] = (array_stand [:,i] - array_stand [:,i].mean()) / array_stand [:,i].std()  # each column is being standardized
	# What to return
	if (transform== 'norm'):
		return pd.DataFrame(array_norm, columns= dataframe.columns)
	elif (transform== 'stand'):
		return pd.DataFrame(array_stand, columns= dataframe.columns)
	elif (transform== 'all'):
		return(pd.DataFrame(array_norm, columns= dataframe.columns), pd.DataFrame(array_stand, columns= dataframe.columns))
	else:
		return '=> Error: Invalid entry for argument-transform'





