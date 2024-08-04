
# Data Mining and Data Processing Project

## Project Description
This project is centered on the collection, extraction, processing, and analysis of data from various sources with a specific focus on car price predictions. It encompasses three main phases:
1. Data collection and cleaning.
2. Advanced data analysis and price prediction using machine learning models.
3. Creating a Flask web service and HTML form for user input and car price prediction.

## Features
### Data Collection and Cleaning
- **Data Collection**: Scraping data from web sources like ad.co.il using BeautifulSoup and requests libraries.
- **Data Cleaning**: Handling missing values, removing duplicates, and correcting data types using pandas.

### Data Transformation and Modeling
- **Data Transformation**: Normalizing and transforming data into a usable format for analysis.
- **Data Analysis**: Utilizing Elastic Net regression models to predict car prices based on collected data.
- **Data Visualization**: Creating visualizations with matplotlib and seaborn to represent findings and model performance.
- **Model Training**: Training the model using the prepared data.

### Web Service and User Interface
- **Flask Application**: Creating a Flask web service to handle user input and return predictions.
- **HTML Form**: Developing an HTML form for users to input car details and receive price predictions.

## Technologies Used
- **Python**: Primary programming language used for scripting and data analysis.
- **Pandas**: Used for data manipulation and cleaning.
- **NumPy**: Supports Pandas with numerical functions.
- **BeautifulSoup (bs4)**: Utilized for web scraping to parse HTML and XML documents.
- **Requests**: Handles HTTP requests to fetch data from the web.
- **Matplotlib**: For creating static, interactive, and animated visualizations.
- **Scikit-learn**: Provides tools for data mining and data analysis, including the ElasticNet regression model.
- **Flask**: Micro web framework used to create the web service.
- **HTML/CSS**: For creating the user interface form.
- **StandardScaler, OneHotEncoder**: Preprocessing modules for feature scaling and encoding categorical variables.
- **GridSearchCV, train_test_split**: For optimizing model parameters and splitting data sets.
- **Pipeline**: Simplifies the process of chaining multiple estimators into one.
- **mean_squared_error**: Metric used to evaluate the performance of the regression model.
- **Urllib**: Used for URL handling.
- **re (Regular Expressions)**: Helps in data cleaning by searching for patterns within text.

## Project Structure
```
data-mining-project/
├── app
|   ├── static/
│   |   ├── car-animation.jpeg
│   |   └── LOGO.jpeg
|   ├── templates/
│   |   └── index.html
│   ├── car_data_prep.py
|   ├── model_training.py
|   ├── api.py 
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── cleaned/
│
├── notebooks/
│   ├── Part1_Data_Collection_and_Cleaning.ipynb 
│   └── Part2_Data_Analysis_and_Modeling.ipynb
│
├── models/
│   └── trained_model.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```
- **app/**: Contains the Flask web application and related files.
  - **templates/**: Contains the HTML template for the web application.
  - **static/**: Contains static files such as images for the web application.
  - **car_data_prep.py**: Script for preparing data for model training and prediction.
  - **model_training.py**: Script for training the machine learning model.
  - **api.py**: Flask application to serve the prediction model.
- **data/**: Hosts all datasets in three subfolders:
  - **raw/**: Original data as collected.
  - **processed/**: Data formatted and preliminarily cleaned.
  - **cleaned/**: Fully cleaned data ready for analysis.
- **notebooks/**: Contains two Jupyter notebooks:
  - **Part1_Data_Collection_and_Cleaning.ipynb**: For data collection and initial cleaning.
  - **Part2_Data_Analysis_and_Modeling.ipynb**: For data analysis and model building.
- **models/**: Stores serialized machine learning models, including a pre-trained price prediction model.
- **templates/**: Contains the HTML template for the web application.
- **static/**: Contains static files such as images for the web application.
- **car_data_prep.py**: Script for preparing data for model training and prediction.
- **model_training.py**: Script for training the machine learning model.
- **api.py**: Flask application to serve the prediction model.
- **requirements.txt**: Lists all Python dependencies.
- **README.md**: Project documentation.
- **.gitignore**: Specifies files to be ignored by Git.

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

3. **Run the Flask application**:
    ```bash
    python api.py
    ```

4. **Access the application**:
    Open your web browser and go to `http://127.0.0.1:5000` to use the car price prediction form.

## License
This project is licensed under the CC BY-NC 4.0 License - see the LICENSE file for details.

