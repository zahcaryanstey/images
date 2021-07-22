<h1> Notes </h1> 
<h2> Libraries required to run scripts  </h2>
<p> Pillow </p>
<p> torch </p>
<p> torchvision</p>
<p> rasterio </p>
<p> matplotlib </p> 
<p> pandas </p>
<p> wandb </p>
<p> sklearn </p>
<p> numpy </p> 
<h2> Files:  </h2>
<h3> Cell image  </h3>
<p> This is a file that contains a small sub set of images which was used for testing some of the scripts included in this git hub repo.</p>
<h3> Directories </h3>
<p> This is a word document that contains a list of the directories that I have data saved in. I use this file to keep track of the directories that I need for my code. </p>
<h3> Processing files </h3>
<p> This folder contains scripts that were used for processng the data before it is ran through my deep learning models. Files that are contained in this folder are  </p>
<p> - ChangeFileExtension.py : This is a script that is used to chagne the file extensions of the images from .ome.tif to .tif.</p>
<p> - CAKI2csv.py : This is a script that is used to create a csv file for use as the ground truth in my deep learning models. </p>
<h3> miscellaneous </h3>
<p> This folder contains scripts that do not fit in to any of the other folders but are important in the devlopment of the project.  </p>
<h1> Procedure </h1>
<p> First CAKI2 Images were exported from the IDEAS software as 8bit padded images. </p>
<p> Next,feature files were downloaded from the paper  An image-based flow cytometric approach to the assessment of the nucleus-to-cytoplasm ratio by Joseph Sebastian et al. These feature files were used as the ground truth in my deep learning models. </p>
<p> Next the feature files CAKI2_CellDiameter.txt and CAKI2_Nucleus_Diameter.txt were loaded into python using pandas dataframes. these data frames had column names  |Cell_ObjectNumber||Cell_diameter| and |Nucleus_ObejctNumber||Nucleus_diameter|.  The two data frames where then concatenated to form one datframe that had column names |Nucleus_ObjectNumber|Nucleus_diameter|Cellular_ObjectNumber||Cell_diameter|. Next One of the ObjectNumber columns were deleted because it is not necessary to have to object number columns that contain the same infromation, and the columns were reanmes so that there names were now |cell_diameter||Nucleus_diameter||Object_Number|. Next the colums were converted to floats so that we can do math with the columns. Then a new column was added called NC where the Nucleus diaemter was divided by the Cell diameter so now that column names are |object number||cell_diameter|nucleus_diameter||NC|. Finaly a column was added called file name which contained all thie file names this was done by turining the object number column into a string and adding .tif at the end of the object numbers. The final table has the form of |File_Name||Object_Number||Cell_Diameter||Nucleus_Diameter||NC|</p>
