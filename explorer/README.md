# Band Structure Exploration
This folder allows you to easily explore the band structures on a locally hosted web app, usually at http://127.0.0.1:8050/. To run, simply navigate to this folder and install the required libraries by running 

 <pre><code>pip install -r requirements.txt </code></pre> 

 After installing, you can call
 <pre><code>python explorer.py </code></pre>

to run the app. Once inside, a popup window will provide an explanation on how to use the app to search band structures. Finally, if our server is not down the website should also be hosted at https://bands.heuslerene.com. 

Note, if you aree not in this folder when calling the run command, the file systems will not work and the application will crash.

# File Structure
- **explorer/** contains explorer.py which allows you to examine different clusters and .csv files of fingerprints and umap projections.
    - **fingerprints/** contains .csv files for each of the encoders and their umap projection.
    - **encode_fingerprint.ipynb** contains code that makes the .csv files
    - **explorer.py** contains the dash web app.