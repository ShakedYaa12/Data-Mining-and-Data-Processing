
**Important note:** The project is not finished yet.

# Data Mining and Data Processing


## Project Description
This project is centered on the collection, extraction, processing, and analysis of data from various sources with a specific focus on car price predictions. It encompasses two main phases: data collection and cleaning, followed by advanced data analysis and price prediction using machine learning models.


## Features
- **Data Collection**: Scraping data from web sources like ad.co.il using BeautifulSoup and requests libraries.
- **Data Cleaning**: Handling missing values, removing duplicates, and correcting data types using pandas.
- **Data Transformation**: Normalizing and transforming data into a usable format for analysis.
- **Data Analysis**: Utilizing Elastic Net regression models to predict car prices based on collected data.
- **Data Visualization**: Creating visualizations with matplotlib and seaborn to represent findings and model performance.


## Technologies Used
- **Python**: Primary programming language used for scripting and data analysis.
- **Pandas**: Used for data manipulation and cleaning.
- **NumPy**: Supports Pandas with numerical functions.
- **BeautifulSoup (bs4)**: Utilized for web scraping to parse HTML and XML documents.
- **Requests**: Handles HTTP requests to fetch data from the web.
- **Matplotlib**: For creating static, interactive, and animated visualizations.
- **Scikit-learn**: Provides tools for data mining and data analysis, including the ElasticNet regression model.
- **StandardScaler, OneHotEncoder**: Preprocessing modules for feature scaling and encoding categorical variables.
- **GridSearchCV, train_test_split**: For optimizing model parameters and splitting data sets.
- **Pipeline**: Simplifies the process of chaining multiple estimators into one.
- **mean_squared_error**: Metric used to evaluate the performance of the regression model.
- **urllib**: Used for URL handling.
- **re (Regular Expressions)**: Helps in data cleaning by searching for patterns within text.


## Project Structure
```plaintext
data-mining-project/
│
├── data/
│ ├── raw/
│ ├── processed/
│ └── cleaned/
│
├── notebooks/
│   ├── Part1_Data_Collection_and_Cleaning.ipynb 
│   └── Part2_Data_Analysis_and_Modeling.ipynb
│
├── models/
│ └── price_prediction_model.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```
This project is organized to facilitate easy navigation and efficient data management:

- **data/**: Hosts all datasets in three subfolders:
  - **raw/**: Original data as collected.
  - **processed/**: Data formatted and preliminarily cleaned.
  - **cleaned/**: Fully cleaned data ready for analysis.

- **notebooks/**: Contains two Jupyter notebooks:
  - **Part1_Data_Collection_and_Cleaning.ipynb**: For data collection and initial cleaning.
  - **Part2_Data_Analysis_and_Modeling.ipynb**: For data analysis and model building.

- **models/**: Stores serialized machine learning models, including a pre-trained price prediction model.


## How to Run
1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/data-mining-project.git
    cd data-mining-project
    ```
2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the Jupyter notebooks** to perform data collection, cleaning, analysis, modeling, and visualization.

## License
This project is licensed under the CC BY-NC 4.0 License - see the LICENSE file for details.



