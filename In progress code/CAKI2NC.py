import pandas as pd

# Paths to nucleus and cell txt files that contain there respective cell numbers and diameters
path_Nucleus = '/home/zachary/Desktop/Masters Degree /CSV files/CAKI2/CAKI2_Nucleus_Diameter2.txt'
path_Cell ='/home/zachary/Desktop/Masters Degree /CSV files/CAKI2/CAKI2_Cell_Diameter2.txt'


# Read the nucleus txt file and then print the first 5 rows
Nucleus = pd.read_csv(path_Nucleus, sep='\t',header=None,names=['Nucleus_Object_Number','Nucleus_Diameter'])[2:]
print(Nucleus.head())


# Read the cell txt file and then print the first 5 rows
Cell = pd.read_csv(path_Cell, sep='\t',header=None,names=['Cell_Object_Number','Cell_Diameter'])[2:]
print(Cell.head())


# Add both the Nucleus and Cell txt files together into one table and print the first 5 rows
data = pd.concat([Cell,Nucleus],axis = 1)
print(data.head())

# drop the Nucleus object number class because we only need to have one object number column
data = data.drop(columns=['Nucleus_Object_Number'])
print(data.head())

# Change each column to floats so that we can do math with them
data['Cell_Object_Number'] = data['Cell_Object_Number'].astype('float')
data['Cell_Diameter'] = data['Cell_Diameter'].astype('float')
data['Nucleus_Diameter'] = data['Nucleus_Diameter'].astype('float')
print(data.head())


# Rename the columns
data.columns = ['Object_Number','Cell_Diameter','Nucleus_Diameter']
print(data.head())

# Now to clean the data. First we have to remove cells that have a nucleus diameter != 0
# and then remove cells that have a cell diameter greater than there nucleus diameter
data = data[data.Nucleus_Diameter != 0 ]
data = data[data.Cell_Diameter > data.Nucleus_Diameter]

# Now to define a NC column for our data table
data['NC'] = data['Nucleus_Diameter'] / data['Cell_Diameter']
print(data.head())

# Now to save this new data table as a csv file
data.to_csv('CAKI2NC.csv',index=False)

# read the csv file to make sure that its there
pd.read_csv('CAKI2NC.csv')