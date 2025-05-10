[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://icp-demo.streamlit.app)

# icp-demo
Deployable Streamlit app showing ICP pointcloud registration demo with Open3D library

### Installation
```bash
pip install -r requirements.txt
```
### Run the app
```bash
streamlit run streamlit_app.py
```
### Open the app
Open your browser and go to `http://localhost:8501` to view the app:
![App Screenshot](screenshot.png)

### Usage
1. Choose default demo pointclouds or upload your own
2. Check the input pointclouds in the 3D viewer
3. Check and modify ICP parameters in the dropdown menu
4. Click "Run ICP" to perform the registration
5. See the logs of the convergence of the ICP algorithm
6. View the aligned pointclouds in the 3D viewer

### Public Streamlit App
You can also view the app online at [icp-demo.streamlit.app](https://icp-demo.streamlit.app)