# Output2RegularGrid
A program for converting DFlowFM map output to a regular spaced grid.


**requirements:**
* DFlowFM .map output
* Yaml Conda environment. Windows OS.

# Setting up code for modification and usage
* Clone this repo with 
```bash
git clone https://github.com/openearth/dflowfm_regularize_output.git
```
* In Anaconda Powershell
```bash
conda env create --file environment.yml
conda activate dfm_tools_env
```
# Run the test case
* Copy the DFlowFM output from the P drive, or modify the code to read it from there. The test case is located at ```P:\11203850-coastserv\Handover_from_Wilms\Postprocessing```
* Run the regularisation script:
```bash
python nc2regularGrid_listComprehension.py
```
*This will create an 'output' directory and within in it the stitched regular spaced NetCDF file.


