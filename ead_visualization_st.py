# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:13:16 2021

@author: eadob
"""

### Importing Relevant Packages
import numpy as np
import streamlit as st
import math

from sklearn import model_selection
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from mlxtend.plotting import plot_decision_regions


################################################################################################

# Plotting decision regions
#@st.cache(suppress_st_warning=True)
def decision_boundary_plot(model, mdl_name, dataframe, y_val, x_vals, split_size, rand_state):
    # Select X and Y variables
    X = dataframe.loc[:, x_vals]
    Y = dataframe.loc[:, y_val]
    
    seed = (100 - split_size) / 100
    
    # Split data
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=seed, random_state=rand_state)
    # Encode y values
    uni_vals = sorted(y_train.unique().tolist())
    y_val_en = y_train.replace({val:num for num, val in enumerate(uni_vals)}).copy()
    
    # plot decision boundary for pedal width vs pedal length   
    max_x_axs, min_x_axs = math.ceil(X.iloc[:, 0].max()) * 1.0, math.floor(X.iloc[:, 0].min() * 1) - 0.10
    max_y_axs, min_y_axs = math.ceil(X.iloc[:, 1].max()) * 1.0, math.floor(X.iloc[:, 1].min() * 1) - 0.10
    
    plot_step_x = 0.003 * (max_x_axs - min_x_axs)
    plot_step_y = 0.003 * (max_y_axs - min_y_axs)
    plot_colors =  "rybgm" #['red', 'yellow', 'blue', 'green', 'magenta', 'orange','purple', 'cyan', 'gold'] #
    
    xx, yy = np.meshgrid(np.arange(min_x_axs, max_x_axs, plot_step_x), np.arange(min_y_axs, max_y_axs, plot_step_y))
    plt.tight_layout(h_pad=1, w_pad=1, pad=2.5)
    
    # Fitting the model for plotting
    model.fit(x_train, y_val_en)
    pred_all = model.predict(np.c_[xx.ravel(), yy.ravel()])
    pred_all = pred_all.reshape(xx.shape)
    
    graph = plt.contourf(xx, yy, pred_all, cmap=plt.cm.RdYlBu)  # colors=['red', 'yellow', 'blue', 'green', 'magenta', 'orange','purple', 'cyan', 'gold']
    
    plt.xlabel(x_train.columns[0])
    plt.ylabel(x_train.columns[1])
    
    print(x_train.iloc[:, 0].max())
    print(x_train.iloc[:, 1].max())
    
    ## Index for x_test
    ind_x1 = x_test.columns.get_loc(x_train.columns[0])
    ind_x2 = x_test.columns.get_loc(x_train.columns[1])
    
    # plot test data points
    cn = uni_vals  #['setosa','versicolor','virginica']
    for i, color in zip(cn, plot_colors):
        temp = np.where(y_test == i)
        idx = [elem for elems in temp for elem in elems]
        plt.scatter(x_test.iloc[idx, ind_x1], x_test.iloc[idx, ind_x2], c=color, 
                    label=y_test, cmap=plt.cm.RdYlBu, edgecolor='black', s=20)
    plt.suptitle(mdl_name)
    
    
# Plotting decision regions
#@st.cache(suppress_st_warning=True)
def decision_boundary_plot_2(model, mdl_name, dataframe, y_val, x_vals, split_size, rand_state):
    # Select X and Y variables
    X = dataframe.loc[:, x_vals]
    Y = dataframe.loc[:, y_val]
    # Find unique values in Y
    uni_Y = sorted(Y.unique().tolist())
    # Encode unique values in Y using the index of the list by using the enumerate function
    encoding = {val:num for num, val in enumerate(uni_Y)}
    # Replace the Y values using the encoding values
    Y_en = Y.replace(encoding).copy()
    
    seed = (100 - split_size) / 100   # Specifying the % of test as a decimal
    
    # Split data
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=seed, random_state=rand_state)
    #st.info(encoding)
    
    # Fit model
    model.fit(X, Y_en)
    
    # Plotting parameters
    scatter_kwargs = {'s': 50, 'edgecolor': None, 'alpha': 0.7}
    contourf_kwargs = {'alpha': 0.2}
    scatter_highlight_kwargs = {'s': 50, 'label': 'Test data', 'alpha': 0.7}
    
    # Plot boundaries
    ax = plot_decision_regions(X=X.values, y=Y_en.values, clf=model, legend=0, X_highlight=None, scatter_kwargs=scatter_kwargs, 
                               contourf_kwargs=contourf_kwargs, scatter_highlight_kwargs=scatter_highlight_kwargs)  # x_test.values
    # X, Y and Title labels
    plt.xlabel(x_vals[0])
    plt.ylabel(x_vals[1])
    plt.title(mdl_name)
    
    # Relabeling legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, encoding.keys(), framealpha=0.3, scatterpoints=1)
     

# Plot various chart and graphs
#@st.cache(suppress_st_warning=True) 
def data_graph(df, classes=None, plt_type='correlation', mdl_type='classification'):
    if mdl_type.lower() == 'classification':
        colors = classes
    else:
        colors = None
    ### Determine the number of columns
    headers = df.columns
    if len(headers) - 1 == 1:
        cols = 1
        #row = 1
    elif (len(headers) - 1) in [2, 3, 4]:
        cols = 2
        #rows = 1 if (len(headers) - 1) == 2 else 2
    else: 
        cols = 3 
        #rows = math.ceil(headers / 3)
    ### Selecting graph type
    if plt_type.lower() == 'correlation matrix':
        fig1, ax1 = plt.subplots()
        #matrix = np.triu(df.corr())
        ax1 = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG', mask=None, annot_kws={"size":8})
        ax1.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
        ax1.set_xticklabels(ax1.get_xmajorticklabels(), fontsize = 8, rotation=90)
        ax1.set_yticklabels(ax1.get_ymajorticklabels(), fontsize = 8, rotation=0)
        st.pyplot(fig1)
    elif plt_type.lower() == 'scatter matrix':
        fig2, ax2 = plt.subplots()
        ax2 = sns.pairplot(df, hue=colors,kind='scatter', diag_kind='kde')
        st.pyplot(ax2)
    elif plt_type.lower() == 'box & whisker':
        fig3, ax3 = plt.subplots()
        plt.xticks(fontsize=7, rotation=90)
        plt.yticks(fontsize=7, rotation=0)
        ax3 = sns.boxplot(data=df)
        st.pyplot(fig3)
    elif plt_type.lower() == 'parallel coordinates':
        fig4, ax4 = plt.subplots()
        plt.xticks(fontsize=7, rotation=90)
        plt.yticks(fontsize=7, rotation=0)
        parallel_coordinates(df, classes, color= ['#556270', '#4ECDC4', '#C7F464', 'blue', 'red', 'green', 'orange','purple', 
                                                  'magenta', 'cyan', 'gold', '#2e8ad8', '#cd3785', '#c64c00', '#889a00'])  
        st.pyplot(fig4)
    elif plt_type.lower() == 'violin':
        cnt5 = 0
        vio_cols = st.beta_columns(cols)
        for i in range(len(headers) - 1):
            fig5, ax5 = plt.subplots()
            #ax = fig5.add_subplot(int(f'{rows}{cols}1') + i) # 221 + i
            sns.violinplot(x=classes, y=headers[i], data=df, split=True, ax=ax5)
            vio_cols[cnt5].pyplot(fig5)
            cnt5 += 1
            if cnt5 > cols - 1:
                cnt5 = 0
    elif plt_type.lower() == 'histogram':
        cnt6 = 0
        hist_cols = st.beta_columns(cols)
        for i in range(len(headers) - 1):
            fig6, ax6 = plt.subplots()
            #sns.set_theme()
            sns.distplot(df[headers[i]]) # , hist=True, fit=norm
            hist_cols[cnt6].pyplot(fig6)
            cnt6 += 1
            if cnt6 > cols - 1:
                cnt6 = 0        


###### Function to bar plot the results for ML models
#@st.cache(suppress_st_warning=True)
def plot_bar(model, x_lab="Algorithms", y_lab="Accuracy (%)"):
    # X and Y values plotted
    x = model[0] 
    mu = [round(j*100) for j in model[1]]
    std = [round(j*100) for j in model[2]] 
    pred = [round(j*100) for j in model[3]] 
    # Bar width and indices of x labels
    width = 0.37
    x_pos_cv = [i for i, _ in enumerate(x)]
    x_pos_pred = [x + width for x in x_pos_cv]
    # Plotting CV and prediction accuracies
    plt.bar(x_pos_cv, mu, width, color='gray', yerr=std, capsize=3, label='Cross-validation')
    plt.bar(x_pos_pred, pred, width, color='gold', label='Prediction')
    # Setting axes labels and legends
    plt.xlabel(x_lab, fontweight='bold')
    plt.xticks([s - (width/ 2) for s in x_pos_pred], x) #[s / 2 for s in x_pos_pred]
    plt.ylabel(y_lab, fontweight='bold')
    plt.yticks(range(0,110,10))
    plt.legend(loc='best', fontsize=9)
    #plt.title("Energy output from various fuel sources")
    plt.show()




# def decision_boundary_plot_2(model, mdl_name, dataframe, y_val, x_vals, split_size, rand_state):
#     # Select X and Y variables
#     X = dataframe.loc[:, x_vals]
#     Y = dataframe.loc[:, y_val]
    
#     seed = (100 - split_size) / 100
    
#     # Split data
#     x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=seed, random_state=rand_state)
#     # Encode y values
#     uni_vals = sorted(y_train.unique().tolist())
#     y_val_en = y_train.replace({val:num for num, val in enumerate(uni_vals)}).copy()
    
#     # plot decision boundary for pedal width vs pedal length   
#     max_x_axs, min_x_axs = math.ceil(X.iloc[:, 0].max()) * 1.0, math.floor(X.iloc[:, 0].min() * 1) - 0.10
#     max_y_axs, min_y_axs = math.ceil(X.iloc[:, 1].max()) * 1.0, math.floor(X.iloc[:, 1].min() * 1) - 0.10
    
#     plot_step_x = 0.003 * (max_x_axs - min_x_axs)
#     plot_step_y = 0.003 * (max_y_axs - min_y_axs)
#     plot_colors =  "rybgm" #['red', 'yellow', 'blue', 'green', 'magenta', 'orange','purple', 'cyan', 'gold'] #
    
#     xx, yy = np.meshgrid(np.arange(min_x_axs, max_x_axs, plot_step_x), np.arange(min_y_axs, max_y_axs, plot_step_y))
#     plt.tight_layout(h_pad=1, w_pad=1, pad=2.5)
    
#     # Fitting the model for plotting
#     model.fit(x_train[x_vals], y_val_en)
#     # model.fit(x_train, y_val_en)
#     # pred_all = model.predict(np.c_[xx.ravel(), yy.ravel()])
#     # pred_all = pred_all.reshape(xx.shape)
    
#     N = 300
#     X = np.linspace(min_x_axs, max_x_axs, N)
#     Y = np.linspace(min_y_axs, max_y_axs, N)
#     X, Y = np.meshgrid(X, Y)
    
#     test_data = x_test
#     test_data[y_val] = y_test
    
#     g = sns.FacetGrid(test_data, hue=y_val, height=5, palette = 'colorblind').map(plt.scatter,x_vals[0], x_vals[1], ).add_legend()
#     my_ax = g.ax
    
#     zz = np.array([model.predict(np.array([[xx,yy]])) for xx, yy in zip(np.ravel(X), np.ravel(Y)) ] )
#     Z = zz.reshape(X.shape)
    
#     # Plot the filled and boundary contours
#     my_ax.contourf( X, Y, Z, 2, alpha = .1, colors = ('blue','green','red'))  # , colors = ('blue','green','red')
#     my_ax.contour( X, Y, Z, 2, alpha = 1, colors = ('blue','green','red'))    # , colors = ('blue','green','red')
    
#     #Add axis and title
#     my_ax.set_xlabel(x_vals[0])
#     my_ax.set_ylabel(x_vals[1])
#     my_ax.set_title(mdl_name)

