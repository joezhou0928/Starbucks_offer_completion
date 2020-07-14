# Starbucks_offer_completion
### Summary:
This project is about disaster messages classification based on machine learning. During each disaster event, people would post messages on different categories depending on their urgent needs. Knowing the category/categories of each message can improve the efficiency of disaster relief agencies' work. An web app is also built to visualize the analysis.
### Files:
There are three folders: app, data and models.
1. **data**: Two csv files contain the data we need. process_data.py will process raw data and write clean data into a database. This is the *first* python file to execute.
2. **models**: train_classifier.py will read in the clean data from the database and train a machine learning for the classification. This is the *second* python file to execute.
3. **app**: run.py file will set up the application for us as the last step. This is the *last* python file to execute.
### Instructions:
1. Run the following commands to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database (needed files are under the folder called **data**)
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves (needed file is under the folder called **models**)
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to https://SPACEID-3001.SPACEDOMAIN to check your application! To know the SPACEID and SPACEDOMAIN, type in 'env | grep WORK' in your command prompt.
### Examples:
**Image 1**: Message Classification
![Message Classification](https://github.com/joezhou0928/Disaster-message-classification/blob/master/ML.png)
**Image 2**: Distribution of Total Number of Categories Related
![Distribution of Total Number of Categories Related](https://github.com/joezhou0928/Disaster-message-classification/blob/master/Viz.png)
