# Urban City Analysis ML

**Urban City Analysis ML** is a comprehensive, menu-driven data science solution for analyzing urban city data using machine learning. The project enables you to predict crime occurrence, accident severity, and passenger count from real-world urban datasets with detailed explanations and user-friendly interface.

---

## 🚀 Main Features

### **📊 Comprehensive Data Analysis Pipeline**
- **Smart Data Preprocessing:** Automatically handles missing values, encodes categorical features, and scales numerical data
- **Detailed Statistics Display:** Shows dataset overview with unique values, ranges, and data quality metrics
- **Exploratory Data Analysis:** Interactive visualizations including box plots, correlation heatmaps, and scatter plots

### **🎯 Three ML Analysis Options**

#### **1. 🚨 Crime Prediction Analysis**
- **Purpose:** Predicts likelihood of crime occurrence based on urban environmental factors
- **Algorithm:** Ensemble stacking model combining Random Forest and Gradient Boosting
- **Features Used:** 
  - Passenger count and trip duration
  - Geographic location (district, latitude, longitude)
  - Weather conditions and vehicle count
  - Time-based features (hour, day of week, weekend indicator)
- **Output:** Accuracy metrics, feature importance, confusion matrix, and classification report

#### **2. 🚗 Accident Severity Prediction**
- **Purpose:** Classifies accident severity levels (low/medium/high)
- **Algorithms:** Multiple classification models (Random Forest, Decision Tree, Logistic Regression)
- **Features Used:** Weather conditions, vehicle count, location data, traffic factors
- **Output:** Comparative model performance, best model selection, and detailed metrics

#### **3. 🚌 Passenger Count Prediction**
- **Purpose:** Estimates passenger count using advanced regression analysis
- **Algorithm:** Random Forest Regression with hyperparameter tuning via GridSearchCV
- **Features Used:** Weather conditions, time factors, stop locations, accident severity, crime category
- **Advanced Features:** Feature engineering with interaction terms and cross-validation
- **Output:** R² scores, MSE, RMSE, MAPE, and feature importance rankings

### **💫 User Experience Features**
- **Menu-Driven Interface:** Easy navigation with colored, descriptive menu options
- **Progress Indicators:** Real-time feedback with emojis and colored status messages
- **Detailed Explanations:** Each analysis includes methodology descriptions and result interpretations
- **Error Handling:** Graceful handling of missing data and user input validation
- **Optional Visualizations:** Choose whether to display data exploration charts

---

## 🛠️ How to Run

### **1. Clone the Repository**
```bash
git clone https://github.com/dharm1123/Urban-City-Analysis-ML.git
cd Urban-City-Analysis-ML
```

### **2. Install Required Libraries**
```bash
pip install -r requirements.txt
```

**Main Dependencies:**
- pandas (data manipulation)
- numpy (numerical computations)
- scikit-learn (machine learning algorithms)
- matplotlib & seaborn (data visualization)
- termcolor (colored terminal output)

### **3. Prepare the Dataset**
- Ensure `final_crime_dataset.csv` is in the project root directory
- For passenger count analysis, `passenger_count_dataset_modified.csv` is also recommended
- The system will automatically handle missing files and adapt accordingly

### **4. Run the Main Program**
```bash
python PROJECTEND.py
```

### **5. Follow the Interactive Menu**
1. **Data Loading:** The program automatically loads and preprocesses your data
2. **Visualization Choice:** Decide whether to see exploratory data analysis charts
3. **Analysis Selection:** Choose from:
   - `1` - Crime Prediction Analysis
   - `2` - Accident Severity Prediction  
   - `3` - Passenger Count Prediction
   - `0` - Exit
4. **Results Review:** Examine detailed performance metrics and interpretations

---

## 📁 File Structure

```
Urban-City-Analysis-ML/
├── PROJECTEND.py                          # 🎯 Main program (menu-driven interface)
├── final_crime_dataset.csv               # 📊 Primary dataset (user provided)
├── passenger_count_dataset_modified.csv  # 🚌 Passenger count dataset (optional)
├── requirements.txt                       # 📦 Python dependencies
├── README.md                             # 📚 This documentation
├── About project.docx                    # 📄 Project overview document
├── model.docx                            # 📈 Model documentation
└── [Additional model scripts]            # 🔧 Individual analysis scripts
```

---

## 🔬 Technical Details

### **Data Preprocessing Pipeline**
1. **Missing Value Handling:** Smart imputation using random sampling from existing values
2. **Feature Engineering:** Time-based features, interaction terms, categorical encoding
3. **Data Scaling:** StandardScaler for numerical features, one-hot encoding for categorical
4. **Data Quality Checks:** Outlier detection and removal, data type validation

### **Model Performance Metrics**
- **Classification:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Regression:** R² Score, MSE, RMSE, MAPE, Cross-validation scores
- **Feature Analysis:** Feature importance rankings and interpretations

### **Advanced Features**
- **Ensemble Methods:** Stacking classifier for improved prediction accuracy
- **Hyperparameter Tuning:** GridSearchCV for optimal model parameters
- **Cross-Validation:** Robust model evaluation with multiple data splits
- **Balanced Datasets:** Automatic handling of imbalanced data through sampling

---

## 👥 Authors & Contributors

**Primary Author:** [DHARM DUDHAGARA](https://github.com/dharm1123)  
**Contributors:** 
- [DHARM DUDHAGARA](https://github.com/dharm1123)
- [Deep Isalaniya](https://github.com/Deep-ii)

---

## 📞 Contact & Support

For questions, suggestions, feature requests, or technical support:
- **GitHub Issues:** [Report bugs or request features](https://github.com/dharm1123/Urban-City-Analysis-ML/issues)
- **Repository:** [View source code and documentation](https://github.com/dharm1123/Urban-City-Analysis-ML)

---

## 🌟 Key Improvements in This Version

- ✅ **Menu-driven interface** for better user experience
- ✅ **Comprehensive explanations** for each analysis type
- ✅ **Error handling and validation** for robust operation
- ✅ **Detailed progress indicators** with colored output
- ✅ **Advanced feature engineering** and model optimization
- ✅ **Professional documentation** with clear methodology descriptions
- ✅ **Modular code structure** for easier maintenance and extension

---

*Built with ❤️ for urban data science and smart city analysis*
