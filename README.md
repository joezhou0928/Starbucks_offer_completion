# Starbucks_offer_completion
### Summary:
Once in a while, Starbucks sends out offers to their customers as a marketing strategy. Offers are of different types and every customers may receive different kinds and numbers of special offers. Specifically for this project, we are going to focus on building a **classification machine learning model**. As we know, Starbucks sends out offers with costs, so they are interested in knowing whether one offer will be completed or not by a certain user. If one offer is predicted to be 'not completed' in the future, then Starbucks can save money by not sending out this offer. 

### Link to Medium:
I have a post related to this project here: https://medium.com/@zuobeizhou/who-will-complete-a-starbucks-offer-56e879d22451.

### Files:
There are two folders: data and code.
1. **data**: It contains a data.zip file. There are three .json files within this zip files which are the data sets for this project.
2. **code**: There are two files. One is the a jupyter notebook and the other one is the .py file exported from the notebook.

### Instructions:
1. Download all the files and put them under the same folder.
2. Within that folder, unzip the data.zip file and put all .json files under a folder called 'data'.
3. From here, you are good to go and are able to execute the python code without error. But the code may take a while (40 minutes and more) to run.

### Libraries used:
Please make sure you have these libraries pre-installed: pandas, numpy, math, json, time, matplotlib, seaborn, sklearn and itertools.

### Result:
1. Classification accuracy. The second model achieves an accuracy of 77%. This model is a refined version of the first model with accuracy of 70%.
2. Feature importances. User demographics data are helpful in making predictions including income, age and the time they became a member. However, the amount of money each customer has spent before is the most important feature out of all.


### Result Screenshots from the second model:
**Image 1**: Completion Prediction
![Completion Prediction](https://github.com/joezhou0928/Starbucks_offer_completion/blob/master/completion_prediction.png)

**Image 2**: Feature Importance
![Feature Importance](https://github.com/joezhou0928/Starbucks_offer_completion/blob/master/feature_imp.png)

### Acknowledgement:
*A Collection of Data Science Take-Home Challenges* by Giulio Palombo
