# Urban City Analysis ML

**Urban City Analysis ML** is an end-to-end data science solution for analyzing urban city data using machine learning. The project enables you to predict crime occurrence, accident severity, and passenger count from real-world urban datasets. The main workflow is managed through `PROJECTEND.py`, which provides a menu-driven interface for all key analyses and visualizations.

---

##  Main Features

- **Comprehensive Data Preprocessing:**  
  Cleans, fills missing values, encodes, and scales features for robust modeling.
- **Exploratory Data Analysis & Visualization:**  
  Interactive plots (boxplots, scatter plots, correlation heatmaps, histograms) to understand trends and relationships.
- **Task Selection via Menu:**  
  - **Crime Prediction:** Uses ensemble (stacking) models to predict if a crime occurred.
  - **Accident Severity Prediction:** Multi-class classification of accident severity (low/medium/high) with multiple ML models.
  - **Passenger Count Prediction:** Estimates passenger count using advanced regression and feature engineering.
- **Model Evaluation:**  
  Prints accuracy, confusion matrix, classification report, feature importances, and regression metrics.
- **User-Friendly:**  
  Select your analysis via simple console input.

---

##  How to Run

1. **Clone the Repository**
    ```bash
    git clone https://github.com/dharm1123/Urban-City-Analysis-ML.git
    cd Urban-City-Analysis-ML
    ```

2. **Install Required Libraries**
    ```bash
    pip install -r requirements.txt
    ```
    Main libraries:
    - pandas, numpy, scikit-learn, seaborn, matplotlib, termcolor

3. **Prepare the Dataset**
    - Download or obtain the `final_crime_dataset.csv` (and `passenger_count_dataset_modified.csv` for passenger count task).
    - Place these files in the project root directory.

4. **Run the Main Program**
    ```bash
    python PROJECTEND.py
    ```
    - Enter `1` for Crime Prediction, `2` for Accident Severity Prediction, or `3` for Passenger Count Prediction when prompted.

---

##  File Structure

```
Urban-City-Analysis-ML/
├── PROJECTEND.py              # Main entry point (menu-driven)
├── final_crime_dataset.csv    # Main dataset (user must provide)
├── passenger_count_dataset_modified.csv # (for passenger count task)
├── requirements.txt
├── <other model and utility scripts>
└── README.md
```

##  Author

**DHARM DUDHAGARA**  
[GitHub Profile](https://github.com/dharm1123)

---

##  Contributors

- [DHARM DUDHAGARA](https://github.com/dharm1123)
- [Deep Isalaniya](https://github.com/Deep-ii)

---

##  Contact

For questions, suggestions, or feedback, please use the [GitHub Issues](https://github.com/dharm1123/Urban-City-Analysis-ML/issues) page.

---
