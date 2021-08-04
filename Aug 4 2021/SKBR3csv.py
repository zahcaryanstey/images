"""
This is a script used for the augmentation of the data given in the text files.
For the use of labeling in a deep learning model
we want to table to look like
Cell # | file name | nucelus diameter | cell diatmer | NC |
the steps involved in this process is
1) import pandas
2) read the cell diameter txt file
3) read the nucleus diameter txt file
4) Clean the data
5) Add NC column
6) Add a file name column
7) Rearange the columne names
"""
# 1 import pandas
import pandas as pd


# 2 read the cell diameter txt file
cell = pd.read_csv('/home/zachary/Desktop/DeepLearning/Dataset/SKBR3/Processed/SKBR3_Cell_Diameter2.txt',sep="\t", header=None,names=['Object Number','Diameter'])[2:]

# 3 read the nucleus cell file
nucleus = pd.read_csv('/home/zachary/Desktop/DeepLearning/Dataset/SKBR3/Processed/SKBR3_Nucleus_Diameter2.txt',sep="\t", header=None,names=['Object Number','Diameter'])[2:]

# 4 clean the data
SKBR3_data = pd.concat([cell,nucleus],axis = 1)
SKBR3_data.columns = ['Object_Number','Cell_Diameter','Nucleus_Object_Number','Nucleus_Diameter']


SKBR3_data = SKBR3_data.drop(columns=['Nucleus_Object_Number'])

SKBR3_data['Cell_Diameter']= SKBR3_data['Cell_Diameter'].astype(float)
SKBR3_data['Nucleus_Diameter']= SKBR3_data['Nucleus_Diameter'].astype(float)
SKBR3_data = SKBR3_data[SKBR3_data.Nucleus_Diameter != 0 ]
SKBR3_data = SKBR3_data[SKBR3_data.Cell_Diameter > SKBR3_data.Nucleus_Diameter]
SKBR_data = SKBR3_data.reset_index(drop=True)


# 5 Add NC column
SKBR3_data['NC'] = SKBR3_data['Nucleus_Diameter'] / SKBR3_data['Cell_Diameter']

# 6 Add file name column
SKBR3_data['File_Name']  = SKBR3_data['Object_Number']+'.tif'
SKBR3_data['File_Name'] = SKBR3_data['File_Name'].astype(str)
print('Final table with file names')
print(SKBR3_data.head())



# print(CAKI2_data['File_Name'])


columnsTitles = ['File_Name','NC','Object_Number','Cell_Diameter','Nucleus_Diameter']

SKBR3_data = SKBR3_data.reindex(columns=columnsTitles)
# CAKI2_data = CAKI2_data.drop(['Object_Number','Cell_Diameter','Nucleus_Diameter'],axis=1)
print(SKBR3_data.head())

SKBR3_data.to_csv('SKBR3.csv',index=False)
