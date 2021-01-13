# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 22:13:07 2021

@author: eadob
"""

### Importing Relevant Packages
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge, HuberRegressor, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.neural_network import MLPRegressor 

from ead_visualization_st import data_graph, decision_boundary_plot
from ead_ml_tunning_st import model_development

######################### Setting up Interactive Interface  #######################
#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='ML Predictive Modeling App, Beta Vsn',
    layout='wide')

#---------------------------------#
main_header = """Machine Learning Predictive Modeling App: Beta Version 1.0"""
#bgcolor = st.color_picker('Select color:')
html_main_header = '''
<div style="background-color:{}; padding:50px">
<h1 style="color:{}; font-size:50px; text-align:center">{}</h1>
</div>
'''

st.markdown(html_main_header.format("#078E61", "white", main_header), unsafe_allow_html=True)

col1, mid, col2 = st.beta_columns([2.5,10,2.5])
with col1:
    st.write('\n ###')
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRs2R2RNHAJAhQbW7NXlbgRSBmf2QicIa5cYw&usqp=CAU", use_column_width=True)
    st.image("https://symbolikon.com/wp-content/uploads/edd/2019/09/Adinkra-akomantoaso-bold-400w.png", use_column_width=True)
with mid:
    st.image("https://www.hartenergy.com/sites/default/files/image/2019/02/predictiveanalyticswordcloud.jpg", width=500, use_column_width=True)
with col2:
    st.write('\n ###')
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/ff/Wawa_aba.png", use_column_width=True)
    st.image("https://symbolikon.com/wp-content/uploads/edd/2019/09/Adinkra-nsaa-bold-400w.png", use_column_width=True)

st.info('**General and Technical Information**')

with st.beta_expander("Developer and Development"):
    st.write("""
    **Developer:** Emmanuel A. Dadzie | eadobolous@gmail.com\n""")
    
    st.image("EAD_Photo.jpg", width=300)
    
    st.write("""
    **Development:** This app was developed using numerous packages and modules from the python programming language.
    The beta version 1.0 offers ML algorithms with *default settings* for most of the parameter and hyperparameters
    as programmed in the packages. Users do not have access to change these parameters in this version. The *version
    2.0* will include this and other features to improve the usability of the app. Further upgrades will be deployed
    to include advanced features for higher level modeling.
    """)

with st.beta_expander("Classification and Regression Models"):
    st.write("""
    This app provides 12 different classifiers and regressors derived from over 12 ML algorithms. 
    They are K-Nearest Neighbor **(KNN)**, Classification and Regression Trees **(CART)**, Naive Bayes 
    **(NB)**, Support Vector Machines **(SVM)**, Gradient Boost **(GB)**, AdaBoost **(AB)**, Random Forest 
    **(RF)**, Linear and Logistic Regression **(LR)**, Artificial Neural Nets **(ANN)**, Gaussian Process 
    **(GP)**, Linear and Quadratic Discriminant Analysis **(LDA and QDA)**, Huber Regression **(HR)**, and Extra Trees **(ET)**.
    """)
    
with st.beta_expander("Performance Evaluation Metrics"):
    st.write("""
    This app evaluates the ML models using six (6) different *performance metrics* for each modeling type 
    (classification and regression). The metrics for classification indlude model accuracy **(accuracy)**, f1 
    and weighted f1 score **(f1 and f1_weighted)**, ROC area under the curve **(roc_auc)**, **recall**, and 
    **precision**. Metrics for regression models in this app include **r2, mean squared error, mean absolute error,** 
    **root mean squared error, mean absolute percentage error, and explained variance**\n #####
    ***Note:*** The f1, roc_auc, recall, and precision metric are for only **binary** classification problems. Using these in a multi-class problem will
    produce either an error or no results.""")
    
with st.beta_expander("Other Modeling Parameters"):
    st.write("""
    Other paramters to set are the **data split ratio** (i.e., percentage of training to testing datasets), and
    the **seed for the randomization** of the dataset before split.
    """)
    
with st.beta_expander("Dataset and Sample Data"):
    st.write("""
    The file format accepted by this app is **csv** with size no more than **200 MB**. The dataset must have *column headings*
    in the first row to be used as columns or variable names by this app. When user data is imported, the full features (including
    changing Y and X variables as desired by the user) of this version 1.0 is made available. The **sample dataset** (i.e., the Iris dataset)
    provided by the app has the Y and the X variables *fixed* and analysis limited to only classification. Also, the visualization of the data
    is not activated, and the input variables for the boundary plot is limited to sepal length and width.
    """)
    
with st.beta_expander("Data Exploration and Visualization"):
    st.write("""
    The exploratory data analysis **(EDA)** for this app includes a brief preview of the imported dataset, stats
    on all the columns of the data as well as *missing and zero values*. these details are immediatly displayed
    right after the data is imported, or the sample data button is clicked. This section also provides input fields
    for the selection of the *Y variable and the X variable(s)* used in the visualization and the model building.
    The data visualization provides six (6) different grapghs: **correlation matrix, scatter matrix, box plot, 
    parallel coordinates, violin, and histogram,** which are selected with the aid of a **slider**. The grapghs 
    automatically update based on the changes made in the X and Y variables.""")

with st.beta_expander("Model Building and Output"):
    st.write("""
    This section involves first, the selection of the type of modeling (i.e., either classification or regression) 
    using the **radio buttons** from the sidebar. Next, select the algorithm(s) to be used for the model building using
    the **checkbox**. *At least one algorithm* must to be selected to produce and output. Other parameters such as the
    *performance metrics, random state, and the train-test split* for the dataset can be modified or set.\n ####
    
    For the ouput, in the case of *classification*, **a summary results, graphs of confusion matrices, classification
    reports and decsion boundary plots** are produced for all the algorithms selected. There are **download links** below
    the sumary results and the reports; the graphs are downloaded by **right click and save image** as prefered. The decision 
    boundary plot provides a selection field/box for the selection of a pair of X variables to be ploted. Exactly *two (2) numeric
    variables* must be selected to avoid error pop-ups. In the case of *regression*, **a summary results, bar chart of the reults, 
    and regression report** are produced for each algorithm. The download approach is the same as the above-mentioned.
    .""")

st.markdown('\n## \n##')


#---------------------------------#
# Sidebar - Collects user input features into dataframe
st.sidebar.header('Upload your CSV data')
file_upload = st.sidebar.file_uploader("Upload your input CSV file", type=['csv'])
st.sidebar.markdown("""
[Example CSV input file - Iris](https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv)
""")

### Select modeling type
st.sidebar.header('Modeling Type and Algorithms')
model_type = st.sidebar.radio('Select Model', options=['Classification', 'Regression'])


### Select algorithms
st.sidebar.write('Select Algorithm(s)')
# Put options in multiple columns
col1, col2, col3 = st.sidebar.beta_columns(3)
# Create variables for checkboxes
if model_type.lower() == 'classification':
    knn = col1.checkbox('KNN', key='knn')
    cart = col2.checkbox('CART', key='cart')
    nb = col3.checkbox('NB', key='nb')
    svm = col1.checkbox('SVM', key='svm')
    gb = col2.checkbox('GB', key='gb')
    ab = col3.checkbox('AB', key='ab')
    rf = col1.checkbox('RF', key='rf')
    lr = col2.checkbox('LR', key='lr')
    ann = col3.checkbox('ANN', key='ann')
    gp = col1.checkbox('GP', key='gp')     ## Guassian Process Classifier
    lda = col2.checkbox('LDA', key='lda')
    qda = col3.checkbox('QDA', key='qda')
    # Define a dictionary for the checkboxes 
    algm_dict = {'KNN':knn, 'CART':cart, 'NB':nb, 'SVM':svm, 'GB':gb, 'AB':ab, 'RF':rf, 'LDA':lda, 'LR':lr, 'QDA':qda, 'ANN':ann, 'GP':gp}
elif model_type.lower() == 'regression':
    knn = col1.checkbox('KNN', key='knn')
    cart = col2.checkbox('CART', key='cart')
    nb = col3.checkbox('NB', key='nb')
    svm = col1.checkbox('SVM', key='svm')
    gb = col2.checkbox('GB', key='gb')
    ab = col3.checkbox('AB', key='ab')
    rf = col1.checkbox('RF', key='rf')
    lr = col2.checkbox('LR', key='lr')     ## LogisticRegressionCV or LinearRegression
    ann = col3.checkbox('ANN', key='ann')  ## MLPRegressor
    gp = col1.checkbox('GP', key='gp')     ## GaussianProcessRegressor
    hr = col2.checkbox('HR', key='hr')     ## BaggingRegressor
    et = col3.checkbox('ET', key='et')     ## ExtraTreeRegressor
    # Define a dictionary for the checkboxes 
    algm_dict = {'KNN':knn, 'CART':cart, 'NB':nb, 'SVM':svm, 'GB':gb, 'AB':ab, 'RF':rf, 'LR':lr, 'ANN':ann, 'GP':gp, 'ET':et, 'HR': hr}


# Adding the checkboxes with values == True (ie those selected)
algm_list = [key for key in algm_dict.keys() if algm_dict[key]==True]
    
#st.write('Selected algorithms:', algm_list)

### Sidebar - Specify parameter settings
st.sidebar.header('Set Parameters')
# Select type of metric
if model_type == 'Classification':
    score_metric = st.sidebar.selectbox('Select Metric', options=['accuracy', 'f1', 'f1_weighted', 'roc_auc', 'recall', 'precision'])
else:
    score_metric = st.sidebar.selectbox('Select Metric', options=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_root_mean_squared_error', 'neg_mean_absolute_percentage_error', 'explained_variance'])

split_size = st.sidebar.slider('Select % of training set', 10, 90, 80, 5)
rand_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)

### All Classifiers and Regressors
# Parameters
n_trees = 100
l_rate = 0.1
m_depth = None
# All models 
models_clf = {'KNN' : KNeighborsClassifier(),
  'CART': DecisionTreeClassifier(),
  'NB'  : GaussianNB(),
  'SVM' : SVC(gamma='auto'),
  'GB'  : GradientBoostingClassifier(loss='deviance', learning_rate=l_rate, n_estimators=n_trees, max_depth=m_depth),
  'AB'  : AdaBoostClassifier(n_estimators=n_trees, random_state=rand_state),
  'LDA' : LinearDiscriminantAnalysis(),
  'LR'  : LogisticRegression(),
  'RF'  : RandomForestClassifier(),
  'QDA' : QuadraticDiscriminantAnalysis(),
  'GP'  : GaussianProcessClassifier(1.0 * RBF(1.0)),
  'ANN' : MLPClassifier(alpha=1, max_iter=1000)
  }
models_reg = {'KNN' : KNeighborsRegressor(),
  'CART': DecisionTreeRegressor(),
  'NB'  : BayesianRidge(),
  'SVM' : SVR(C=1.0, epsilon=0.2),
  'GB'  : GradientBoostingRegressor(learning_rate=l_rate, n_estimators=n_trees),
  'AB'  : AdaBoostRegressor(n_estimators=n_trees, random_state=rand_state),
  'RF'  : RandomForestRegressor(),
  'LR'  : LinearRegression(),
  'GP'  : GaussianProcessRegressor(kernel=(DotProduct() + WhiteKernel())),
  'ANN' : MLPRegressor(max_iter=500, learning_rate='constant'), ## 'adaptive', 'constant' by default
  'ET'  : ExtraTreesRegressor(n_estimators=100),
  'HR'  : HuberRegressor() 
  }
# Selecting the checked algorithms
models = {}
if model_type.lower() == 'classification':
	scoring = score_metric
	for name in algm_list:
		models[name] = models_clf[name]
  
elif model_type.lower() == 'regression':
	scoring = score_metric
	for name in algm_list:
		models[name] = models_reg[name]
else: '=> Invalid input for argument: "output"'


#---------------------------------#
# Main panel

#---------------------------------#
if file_upload is not None:
    # Displays the dataset
    st.warning('**1.0** **Explore** the dataset')
    df = pd.read_csv(file_upload)
    st.write(df.head(10))
    # Shape of data
    st.write('*Dimensions of dataset*')
    st.info(f'{df.shape[0]} **rows**,  {df.shape[1]} **columns**')
    
    st.write('**Stats of dataset**')
    st.write('*Descriptive stats*')
    st.write(df.describe())
    df_more_stats = pd.DataFrame(df.isnull().sum(), columns=['Missing_Values'])
    df_more_stats['Zeros'] = df.isin([0]).sum()
    df_more_stats['SKEWNESS'] = df.kurtosis()
    df_more_stats['KURTOSIS'] = df.kurtosis()
    st.write('*Missing Values, Zeros, Skewness, and Kurtosis*',df_more_stats)
    
    ### Dataset and Preprocessing
    st.write('**Select Y variable:**')
    y_val = st.selectbox('', options = df.columns)
    st.write('**Select X variables:**')
    x_vals =  list(df.columns)
    x_vals.remove(y_val)
    sel_x_vals = st.multiselect('', options= x_vals, default=x_vals)
    st.markdown('#\n')
    
    ### Graphs from final modeified dataset
    st.warning('**2.0 Visualize** the Dataset')
    graph_sel = st.select_slider('Select plot', options=['correlation', 'scatter', 'box', 'parallel_coordinates', 'violin', 'histogram'])
    try:
        graph_list = data_graph(df[sel_x_vals + [y_val]], classes=y_val, plt_type=graph_sel, mdl_type=model_type) 
    except ValueError as ve1:
        st.error(f'''**Input Error:** {ve1}\n
                 X variables cannot be a STRING. Remove input of X variable(s) with string values.
                ''')
    st.markdown('#\n')
    
    ### Build the models
    st.warning('**3.0 Build** the ML Models')
    st.info(f'''**Modeling Input**\n
            a) Modeling type       : {model_type}\n
            b) Selected algorithms : {algm_list}\n
            c) Performance meteric : {score_metric}\n
            d) Data split          : {split_size}% training, {100 - split_size}% testing\n
            e) Seed number         : {rand_state}\n
            f) Y variable          : {y_val}\n
            f) X variable(s)       : {sel_x_vals}\n
            ''')
    b1 = st.button('Run Model(s)', key='1')
    #b2 = st.button('')
    #if st.button('Run Model(s)'): 
    if b1:
        try:
            model_development(df, y_val, sel_x_vals, split_size, rand_state, scoring, algm_models= models, sel_algms=algm_list, out_put=model_type)
        except ValueError as ve:
            if 'continuous' in str(ve):
                st.error(f'''**Input Error:** {ve}\n
                1. Y variable must be a set of CATEGORICAL values for classification models. Current input is FLOAT.\n
                2. Any plot below is INVALID due to incorrect input.''')
            elif 'convert string' in str(ve):    
                st.error(f'''**Input Error:** {ve}\n
                1. X variables cannot be a STRING. Remove input of X variable(s) with string values.\n
                2. Any plot below is INVALID due to incorrect input.''')
        except TypeError as te:
            if 'unsupported operand' in str(te):    
                st.error(f'''**Input Error:** {te}\n
                1. Y variable must be a set of FLOAT or INTEGER values for regression models. Current input variable has STRING values.\n
                2. Any plot below is INVALID due to incorrect input.''')

        
        ####################### Decision Boundaries ###############################
    if model_type.lower() == 'classification':
        st.success('**Decision Boundary Plots** of Model(s)')
        st.write('**Select *ONLY* two variables**')
        # Two X variabls selected
        sel_x = st.multiselect(' ', options= list(df.columns), default=list(df.columns[0:2]))
        #if st.button('Generate Plots'):
        b2 = st.button('Generate Plots', key='2')
        if b1 | b2:            
            if len(algm_list) == 1:
                num_cols = 1
            elif len(algm_list) in [2, 3, 4]:
                num_cols = 2
            elif len(algm_list) >= 4:
                num_cols = 3  
            else: num_cols = 3
            cnt3 = 0
            db_cols = st.beta_columns(num_cols)
            for name, model in models.items():
                fig_db, ax_db = plt.subplots(figsize=(5, 3))
                try:
                    ax_db = decision_boundary_plot(model, name, df, y_val, sel_x, split_size, rand_state)
                except ValueError as ve:
                    #if 'data dimension' in str(ve):
                    st.error(f'''**Input Error - {name}:** {ve}\n
                    Exactly 2 variables needed for plots. {len(sel_x)} variables selected.''')
                except IndexError as ie:
                    st.error(f'''**Input Error - {name}:** {ie}\n
                    Exactly 2 variables needed for plots. {len(sel_x)} variables selected. Add another numeric variable.''')
                except TypeError as te:
                    st.error(f'''**Input Error - {name}:** {te}\n
                    Variable cannot have STRING values. Remove variable(s) containing strings.''')
                # Display decision boundaries in plot columns/grid
                db_cols[cnt3].pyplot(fig_db)
                # Display download links download columns/grid               
                cnt3 += 1
                if cnt3 > num_cols - 1:
                    cnt3 = 0
            st.markdown('#\n')
     
else:
    st.info('Awaiting for CSV file to be uploaded.')
    ### Sample Dataset
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    df_sample = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', names=feature_names)

    b1 = st.button('Press to use Example Dataset', key='1')
    if b1:
        # Displays the dataset 
        st.warning('**1.0 Explore** the **Iris** dataset as an example.')
        st.write(df_sample.head(5))

        st.write('*Dimensions of dataset*')
        st.info(f'{df_sample.shape[0]} **rows**,  {df_sample.shape[1]} **columns**')
    
        st.write('*Stats of dataset*')
        st.write(df_sample.describe()) 
        
        st.write('*Y variable*')
        st.info(df_sample.columns[-1])
        st.write('*X variables*')
        st.info(list(df_sample.columns[:-1]))
        
        try:
            model_development(df_sample, df_sample.columns[-1], df_sample.columns[0:-1], split_size, rand_state, scoring, algm_models= models, sel_algms=algm_list, out_put=model_type)
        except ValueError as ve:
            if 'continuous' in str(ve):
                st.error(f'''**Input Error:** {ve}\n
                1. Y variable must be a set of CATEGORICAL values for classification models. Current input is FLOAT.\n
                2. Any plot below is INVALID due to incorrect input.''')
            elif 'convert string' in str(ve):    
                st.error(f'''**Input Error:** {ve}\n
                1. X variables cannot be a STRING. Remove input of X variable(s) with string values.\n
                2. Any plot below is INVALID due to incorrect input.''')
        except TypeError as te:
            if 'unsupported operand' in str(te):    
                st.error(f'''**Input Error:** {te}\n
                1. Y variable must be a set of FLOAT or INTEGER values for regression models. Current input variable has STRING values.\n
                2. Any plot below is INVALID due to incorrect input.''')
        
    
    ###### Decision Boundary Plots
        if model_type.lower() == 'classification':    
            st.success('**Decision Boundary Plots** of Model(s)')
            st.write('**Select *ONLY* two variables**')
            # Two X variabls selected
            #sel_x = st.multiselect(' ', options= list(df_sample.columns), default=list(df_sample.columns[0:2]))
            #if st.button('Generate Plots'):
            #b2 = st.button('Generate Plots', key='2')
            #if b1 | b2:            
            if len(algm_list) == 1:
                num_cols = 1
            elif len(algm_list) in [2, 3, 4]:
                num_cols = 2
            elif len(algm_list) >= 4:
                num_cols = 3  
            else: num_cols = 3
            cnt3 = 0
            db_cols = st.beta_columns(num_cols)
            for name, model in models.items():
                fig_db, ax_db = plt.subplots(figsize=(5, 3))
                try:
                    ax_db = decision_boundary_plot(model, name, df_sample, df_sample.columns[-1], df_sample.columns[0:2], split_size, rand_state)
                except ValueError as ve:
                    #if 'data dimension' in str(ve):
                    st.error(f'''**Input Error - {name}:** {ve}\n
                    Exactly 2 variables needed for plots. {len(sel_x)} variables selected.''')
                except IndexError as ie:
                    st.error(f'''**Input Error - {name}:** {ie}\n
                    Exactly 2 variables needed for plots. {len(sel_x)} variables selected. Add another numeric variable.''')
                except TypeError as te:
                    st.error(f'''**Input Error - {name}:** {te}\n
                    Variable cannot be STRING values. Check input for variables containing strings.''')
                # Display decision boundaries in plot columns/grid
                db_cols[cnt3].pyplot(fig_db)
                # Display download links download columns/grid               
                cnt3 += 1
                if cnt3 > num_cols - 1:
                    cnt3 = 0
            st.markdown('#\n')
            
##############################################################################################





