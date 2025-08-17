# NYC Taxi Sustainability Dashboard 

An interactive **Streamlit dashboard** for analyzing NYC taxi trips, predicting trip distances, and estimating **CO₂ emission savings** from optimized routing using **OSRM (Open Source Routing Machine)** fastest routes.

---

##  Features

- **File upload support**
  - Upload your **trips dataset** (`datasetdata.csv`)
  - Upload **fastest route distances** (`fastestRoutes.csv`)
- **Trip prediction**
  - Predict trip distances using a trained model
  - Compare actual vs optimized (fastest OSRM) routes
- **Scenario analysis**
  - Test different reduction percentages of route distance
  - Explore CO₂ savings under different assumptions
- **Visualizations**
  - Interactive charts with **Altair** and **Plotly**
  - Maps with **pydeck**
- **Sustainability impact**
  - Estimate CO₂ reduction from optimized routes
-**Run :python -m streamlit run src/dashboard.py

---

## 📂 Project Structure

