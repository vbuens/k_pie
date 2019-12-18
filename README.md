# K-PIE
Estimate the percentage of infection in a leaf

## Requirements

cycler==0.10.0\
joblib==0.13.2\
kiwisolver==1.1.0\
matplotlib==3.0.3\
numpy==1.16.3\
opencv-python==4.1.0.25\
Pillow==6.0.0\
pyparsing==2.4.0\
python-dateutil==2.8.0\
scikit-learn==0.21.0\
scipy==1.2.1\
six==1.12.0\
sklearn==0.0\
utils==0.9.0\
webcolors==1.8.1\\

If you get the following error from matplotlib:\
 `ValueError: Format "jpg" is not supported.`\
 Create the file in ~/.matplotlib/matplotlibrc with the following line in it:\
 `backend: TkAgg`\


## Usage

`python perc_infection.py -o output-dir/ -i input-dir -p infected.jpg -n healthy.jpg -k 5 `\
-k = 5 (by default)\
-o = results (by default)
