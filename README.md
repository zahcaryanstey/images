# Required libaries to run scripts:
<p> Pilow 
<p> torchvision
<p> rasterio
<p> matplotlib
<p> os </p>
<h1> Files </h1>
<h2> Cell image file </h2>
<p> File that containes a small sub sebset of images used for the testing of scripts before applying scripts to actual data set.</p>
<h2> Directories </h2> 
<p> word document that contains a list of the dirctories in the actual data set. Note that the directroies that are in bold font are directories that haved been ran through the change file extension script located in the processing files file. </p>
<h2> openimage.py </h2>
<p> Python script used for testing the procedure of opening the images contained in the cell image file after they have been ran through the chagne file extension script. </p>
<h2> Processing files </h2>
<p> This file contains all of the files that are used of the proceesing of the data. the following files are contained in this folder: </p>
<p> - ChangeFileExtension.py : Script used to change the file extensions from .ome.tif to just .tif </p>
<p> - CSV files/ CAKI2 files: In this file you will find two text files which contain feature extractions from the IDEAS software. Aswell as a a csv file which is a procseed version of the two text files. </p>
<p> Test_Image_Split: Folder that contains a sub set of images used to test a srcipt for the splitting of data into testing, trainging and validation sets. 
<h1> Procedure </h1>
<p> Files were exported from the IDEAS software as 8bit non padded images </p>
<p> Images were then ran through the file name changing script so that image names looked like #.tif. </p>
<h1> To do  </h1> 
<p> Try Images exported differnt ways from IDEAS Software.</p>
<p> Start building trainging pipeline </p>
<p> Focus first on channels that were used in the acqusition of the images. </p> 
<p> Make code moduler so that you can specify which channels to use <p> 

