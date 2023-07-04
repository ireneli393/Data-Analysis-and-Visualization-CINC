#!/usr/bin/env python
# coding: utf-8

# In[145]:


import csv
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score

file_path = "/Users/irene/Desktop/datasets/MGH"
file_list = os.listdir(file_path)
file_list.sort()
file_list = file_list[1:]


# In[192]:


#check for if both actual dataset and the predicted results exist
for file in file_list:
    if 'pred' not in file and file[0:-4] + '_pred.csv' not in file_list:
            file_list.remove(file)


# In[196]:


#create a list to store confusion matrix
confusion_matrices = []

class_labels = ['Wake', 'Light', 'Deep','REM sleep']
#create a list to store individual kappa value for each confusion matrix
kappa = []
#create a list to store file name 
file_name = []

n = len(file_list)-1

for num in range(0,n,2):
    #remove '.csv' in the end and store file name for future use
    file_name.append(file_list[num][:-4])
    # read ground truth file
    df_actual = pd.read_csv(f"/Users/user_name/Desktop/datasets/MGH/{file_list[num]}")
    # read predicted values file
    df_pred = pd.read_csv(f"/Users/user_name/Desktop/datasets/MGH/{file_list[num+1]}")
    # concatenate two dataframes based on axis = 1 and drop null value
    df = pd.concat([df_actual, df_pred],axis=1).dropna()
    # add column names to the dataframe
    df.columns = ['actual', 'pred']
    # remove unknown values which is marked as 9 in the ground truth file
    df = df[df['actual'] != 9]
    # standardize the labels
    df['actual'] = df['actual'].replace({2:1,5:3,3:2})
    # get true labels and predicted labels
    true_labels = df.iloc[:, 0]
    predicted_labels = df.iloc[:,1]
    """
    this helps to make sure that each confusion matrix contains 4 labels even though there are fewer
    than 4 labels in either the true labels list or predicted labels list
    """    
    labels = [0, 1, 2, 3]
    #create confusion matrices and store it in the confusion_matrices list
    confusion_matrices.append(confusion_matrix(true_labels, predicted_labels,labels=labels))
    #calculate the cohen kappa value for each confusion matrix and store it in the list
    kappa.append(cohen_kappa_score(true_labels, predicted_labels))


# In[191]:


import matplotlib.pyplot as plt
import numpy as np


colorbar = False
cmap = "Blues"  # Try "Greens". Change the color of the confusion matrix.
## Please see other alternatives at https://matplotlib.org/stable/tutorials/colors/colormaps.html
values_format = ".0f"  # Determine the number of decimal places to be displayed.

# Determine the dimensions of the grid
num_rows = 330
num_cols = 3

# Create a figure and axes for the subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10,600))

# Iterate over the confusion matrices and plot them in the subplots
for idx, cm in enumerate(confusion_matrices):
    # Compute the row and column indices for the subplot
    row = idx // num_cols
    col = idx % num_cols

    # Plot the confusion matrix in the corresponding subplot
    ax = axes[row, col]
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels).plot(
    include_values=True, cmap=cmap, ax=ax, colorbar=colorbar, values_format=values_format)

    # Remove x-axis labels and ticks

    ax.xaxis.set_ticklabels(['', '', '', ''])
    ax.yaxis.set_ticklabels(['', '', '', ''])
    ax.set_xlabel('')
    ax.set_ylabel('')
 

    ax.set_title(f'{file_list[idx*2][:-4]}')


# Add x-labels and x-ticks to the bottom row of subplots

for ax in axes[-1, :]:
    ax.set_xlabel('Predicted Labels')
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# Add y-labels and y-ticks to the leftmost column of subplots
for ax in axes[:, 0]:
    ax.set_ylabel('True Labels')
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_yticklabels(class_labels)

# Adjust the spacing between subplots
plt.tight_layout()

total_subplots = num_rows*num_cols
if len(confusion_matrices) < total_subplots:
    for idx in range(len(confusion_matrices), total_subplots):
        row = idx // num_cols
        col = idx % num_cols
        fig.delaxes(axes[row, col])

# Show the plot
plt.show()



# In[197]:


df_kappa = pd.DataFrame({'file name':file_name,'individual kappa':kappa})
df_kappa.sort_values(by = 'individual kappa', ascending = False)
print("Avg Cohen's kappa:{:.5f}".format(np.mean(kappa)))
df_kappa


# # Compute the sensitivity and specificity of Wake vs Sleep, by converting the problem to two class (club light, deep and REM as sleep). Ignore unknown classes.

# In[107]:


import warnings

# Ignore a specific warning
warnings.filterwarnings("ignore", category=RuntimeWarning)
sensitivity = []
specificity = []
file_name = []
for num in range(0,len(file_list)-1,2):
    file_name.append(file_list[num])
    df_actual = pd.read_csv(f"/Users/irene/Desktop/datasets/MGH/{file_list[num]}")
    df_pred = pd.read_csv(f"/Users/irene/Desktop/datasets/MGH/{file_list[num+1]}")
    df = pd.concat([df_actual, df_pred],axis=1)
    col_names = ['actual', 'pred']
    df.columns = col_names
    df = df.dropna()
    df = df[df['actual'] != 9]
    values_to_replace = [2,3,5]
    replacement_value = 1

    df['actual'] = df['actual'].replace(values_to_replace, replacement_value)
    df['pred'] = df['pred'].replace(values_to_replace, replacement_value)

    true_labels = df.iloc[:, 0].tolist()
    predicted_labels = df.iloc[:,1].tolist()
    labels = [0, 1]
    cm = confusion_matrix(true_labels, predicted_labels,labels=labels)
    TP = cm[1, 1]
    FP = cm[0, 1]
    TN = cm[0, 0]
    FN = cm[1, 0]
    # Compute sensitivity (true positive rate)
    sensitivity.append(TP / (TP + FN))

    # Compute specificity (true negative rate)
    specificity.append(TN / (TN + FP))

df = pd.DataFrame({'file name':file_name,'sensitivity':sensitivity,'specificity':specificity})
df.sort_values(by = 'sensitivity', ascending = False)
print(f"avg sensitivity:{df['sensitivity'].mean()}")
print(f"avg specificity:{df['specificity'].mean()}")
df

