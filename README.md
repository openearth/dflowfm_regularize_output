# Output2RegularGrid
A program for converting DFlowFM map output to a regular spaced grid.


**requirements:**
* DFlowFM .map output
* Yaml conda environment. Windows OS.

# Usage
* Clone this repo with 
```bash
git clone https://github.com/openearth/coastserv_flask_azure.git
```
* In Anaconda Powershell
```bash
conda env create --file environment.yml
```
* Test the container on your local machine first with:
```bash
docker run -p 80:80 -v D:\PROJECTS\2021\COASTSERV_azure\FES\:/app/app/coastserv/static/FES coastserv.azurecr.io/coastserv:v6
```
Where you replace 'D:\PROJECTS\2021\COASTSERV_azure\FES\' with the path to your FES data.
* In a browser, navigate to 'localhost:80' and run a test by using the test.pli file. 
* If all looks well, push the FES data to Azure and deploy your app with:
```bash
./deploy.sh
```
* In a browser, navigate to coastserv.azurewebsites.net. This might take some time to start if it is the first time you're deploying the container.
* If you make any changes to the code on your local machine, push the changes to Azure with redeploy.sh 


