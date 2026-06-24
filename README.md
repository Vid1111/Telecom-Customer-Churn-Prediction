# Telecom Customer Churn Prediction

This is a data science project done as part of our internship at Coding Blocks. The goal was to figure out which telecom customers are likely to leave the company, and then group them into segments so the business can take action before they actually leave.

## Dataset

We used the IBM Telco Customer Churn dataset from Kaggle. It has 7043 rows and 33 columns. Each row is one customer. The dataset has information like how long they have been a customer, what services they use, how much they pay every month, and whether they churned or not.

The target column is Churn Value — 0 means the customer stayed, 1 means they left.

## What we did

**Step 1 — EDA**

We started by exploring the data to understand it. We made several plots to see which factors seem connected to churn. Some things we noticed were that customers on month-to-month contracts churn a lot more than customers on two year contracts. Customers who do not have tech support also tend to leave more. New customers with high monthly bills are the most likely to churn. We also plotted a correlation matrix to see how numerical columns relate to each other. Tenure Months had the strongest negative correlation with churn meaning longer customers are less likely to leave.

**Step 2 — Data Cleaning**

The Total Charges column was stored as text in the Excel file so we converted it to a number. There were 11 rows where Total Charges was empty. We checked those rows and all of them had Tenure Months equal to zero, meaning they were brand new customers who had not been billed yet. So we filled those empty values with zero.

**Step 3 — Dropping columns**

We removed columns that are not useful for prediction. Things like Customer ID, location data, and columns that would only exist after a customer has already churned like Churn Score and Churn Reason. Using those would be data leakage. We also dropped the City column since it had 1129 unique values which would create too many columns after encoding.

**Step 4 — Encoding**

Most columns in the dataset are text like Yes/No or Male/Female. We converted them to numbers using one-hot encoding with pd.get_dummies. We used drop_first=True to avoid multicollinearity.

**Step 5 — Model Training (Baseline)**

We used Random Forest Classifier. The baseline model had 100 trees and no special settings. It got around 78% accuracy but only caught 51% of actual churners which is not good enough for this problem.

**Step 6 — Approach 1 (Handle Class Imbalance)**

We added class_weight balanced to the model. The dataset has 73% customers who stayed and only 27% who churned. Without handling this the model gets lazy and keeps predicting stay. Balanced weighting forces the model to treat churners as more important. Recall went up to around 70%.

**Step 7 — Approach 2 (Hyperparameter Tuning)**

We increased trees to 300 and added max_depth of 10 to stop the trees from growing too deep and memorizing training data. This gave the best overall results across accuracy, recall and f1 score.

**Step 8 — Approach 3 (Feature Importance)**

We checked which features the model relied on the most. Tenure Months and Monthly Charges came out on top. We also plotted a bar chart of the top 10 most important features. Two features at the bottom, Phone Service and Multiple Lines, had almost zero importance so we dropped them and retrained the model to see if it helps. The results were close to Approach 2 but not better so Approach 2 remained the best.

**Step 9 — Confusion Matrix Heatmap**

We plotted the confusion matrix as a heatmap for Approach 2 to make it easier to read. It clearly shows true positives, false positives, true negatives and false negatives in a visual format.

**Step 10 — Hyperparameter Grid Search**

We tested 20 different combinations of tree count and depth to confirm that our chosen settings in Approach 2 were actually the best ones. We sorted results by recall first since catching churners is more important than overall accuracy.

**Step 11 — Model Comparison Table**

We made a clean table comparing all three models side by side showing accuracy, recall, precision and f1 score so it is easy to see which approach improved what.

**Step 12 — Cross Validation**

We used 5-fold cross validation to get a more reliable accuracy and recall score. Normal train-test split can get lucky or unlucky depending on which rows end up in the test set. Cross validation trains and tests on 5 different splits and averages the result so it is more trustworthy.

**Step 13 — ROC AUC**

We plotted the ROC curve and calculated the AUC score. Our model got 0.857 which means it is doing a good job of separating churners from non-churners across all possible thresholds. This is a better metric than accuracy for imbalanced datasets.

**Step 14 — SMOTE**

We tried SMOTE which creates synthetic samples of the minority class to balance the dataset at the data level rather than through weights. However Approach 2 still came out better. This is because class_weight balanced in Approach 2 already handled the imbalance well and adding synthetic data on top of that did not help.

**Step 15 — Learning Curve**

We plotted a learning curve to check if the model is overfitting or underfitting. It shows how training and validation recall change as more training data is added. If the two lines are far apart the model is overfitting. If both are low it is underfitting.

**Step 16 — Business Impact Analysis**

We calculated the actual business impact in terms of revenue. The model catches a certain number of churners per month and each one represents around 65 dollars in monthly revenue. This translates the machine learning results into a real number the business can act on.

**Step 17 — Customer Segmentation**

We used KMeans clustering to group customers into 3 segments based on their tenure, monthly charges, total charges, and predicted churn probability from the model.

We used the elbow method to decide on 3 clusters. We scaled the data first using StandardScaler so that no single column dominates the distance calculation since Total Charges has much larger values than Tenure Months.

The three segments we found were:

- Budget Loyal Customers — medium tenure, low charges, low churn risk around 12%
- High Risk New Customers — new customers, high charges, very high churn risk around 69%
- Loyal Premium Customers — long tenure, highest charges, low to medium churn risk around 23%

**Step 18 — Retention Strategy**

Based on the segments we put together a simple retention strategy table showing what action the business should take for each group. High Risk New customers need immediate attention. Budget Loyal customers just need small rewards. Loyal Premium customers should get VIP treatment to keep them happy.

## Results

Approach 2 with 300 trees and max_depth 10 and class_weight balanced gave the best overall results. The AUC score of 0.857 confirms the model is reliable. We tried SMOTE and feature selection on top of this but neither improved results further which tells us Approach 2 already found a good balance.

From the business impact analysis, catching churners early can save the company a significant amount of revenue every month.

## Files

- CBSOT_DS_Project.ipynb — the main notebook with all code
- Telco_customer_churn.xlsx — the dataset (place in same folder before running)
- README.md — this file

## How to run

1. Download the notebook and the Excel file and keep them in the same folder
2. Open the notebook in Jupyter or Google Colab
3. Run all cells from top to bottom
4. If you get a warning about joblib or CPU cores just ignore it, the code still runs fine

One library you may need to install separately is imbalanced-learn for the SMOTE section. You can install it by running pip install imbalanced-learn in your terminal.

## Libraries used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
