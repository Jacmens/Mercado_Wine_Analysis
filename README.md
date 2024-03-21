
# importing Libraries 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype  
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from tabulate import tabulate
from mpl_toolkits.mplot3d import Axes3D

# dataset path
red_wine = r"C:\Users\isrea\OneDrive\Desktop\G_wine_red_label_2.csv"
black_wine= r"C:\Users\isrea\OneDrive\Desktop\G_wine_black_label_3.csv"

# Reading CSV files
df_red = pd.read_csv(red_wine, delimiter=",")

# Printing column names of the red wine DataFrame
red_wine_columns = df_red.columns
print(red_wine_columns)

# Checking the dataset
red_wine_sample = df_red.iloc[100:150]
print(red_wine_sample)

# converting red wine sample to string()
red_wine_sample_str = red_wine_sample.to_string()

# Adding red wine to the string
red_wine_sample_str += f"\nred wine sample: {red_wine_sample}"

# Write the formatted string to a text file
with open('red wine sample.txt', 'w', encoding='utf-8') as f:
    f.write(red_wine_sample_str)

# Checking data type
red_wine_type = df_red.dtypes
print(red_wine_type)

# Describing dataframe
red_wine_describe = df_red.describe()
print(red_wine_describe)

# converting red wine sample to string()
red_wine_describe_str = red_wine_describe.to_string()

# Adding red wine to the
red_wine_describe_str += f"\nred wine describe: {red_wine_describe}"

# Write the formatted string to a text file
with open('red wine describe.txt', 'w', encoding='utf-8') as f:
    f.write(red_wine_describe_str)

# Missing info
red_wine_info = df_red.isnull().sum()
print(red_wine_info)

# Exploring correlated columns
# Quality Column
sns.set(rc={"figure.figsize":(14,8)})
sns.countplot(x="quality", data=df_red)
plt.xlabel("quality")
plt.ylabel("Count")
plt.title("Count Plot Showing the Distribution of Red_Label_Wine Base on Quality",  fontsize=14, fontweight='bold')

plt.show()

# Computing correlated columns
sns.pairplot(df_red)
plt.title("pairplot Plot Showing likely Combination of Columns in Red_Label_Dataset", fontsize=14, fontweight='bold')

plt.show()

#Computing heatmap
sns.heatmap(df_red.corr(),annot=True,fmt='.2f', linewidths=2)
plt.title("Heatmap Illustrating Correlation Between Columns in the Red_Label_Data", fontsize=14, fontweight='bold')
plt.show()

sns.distplot(df_red['alcohol level'])
plt.title("Distribution Graph Showing Alcohol Concentration with Respect to Quality of the Red_Label_Wine", fontsize=14, fontweight='bold')
plt.show()

# Computing Variations in wine quality with respect to alcohol concentration

sns.boxplot(x='quality', y='alcohol level', data=df_red, showfliers=False)
plt.title("Boxplot Showing Variations in wine quality with respect to alcohol concentration in the Red_Label_Wine_Data", fontsize=14, fontweight='bold')
plt.show()

# correlation between alcohol and PH vslues

sns.jointplot(x='alcohol level', y='PH', data=df_red, kind='reg')
plt.title("Graph Showing correlation between alcohol and PH vslues in the Red_Label_Wine_Data", fontsize=14, fontweight='bold')
plt.show()

# Calculating correlation beween alcohol and PH value

def get_correlation(alcohol, PH, df_red):
    pearson_corr, p_value = pearsonr(df_red[alcohol], df_red[PH])
    print("Correlation between {} and {} is {}".format('alcohol level', 'PH', pearson_corr))
    print("P-value of this correlation is {}".format(p_value))

# Result
get_correlation('alcohol level', 'PH', df_red)

#BLACK_LABEL_WINE OVERVIEW

# Reading CSV files 
df_black = pd.read_csv(black_wine, delimiter=",")

# Printing column names of the black wine DataFrame
black_wine_columns = df_black.columns
print(black_wine_columns)

# Checking the dataset
black_wine_sample = df_black.iloc[100:150]
print(black_wine_sample)

# Checking data type
black_wine_type = df_black.dtypes
print(black_wine_type)

# Describing dataframe
black_wine_describe = df_black.describe()
print(black_wine_describe)

# Missing info
black_wine_info = df_black.isnull().sum()
print(black_wine_info)

# converting black wine sample to string()
black_wine_sample_str = black_wine_sample.to_string()

# Adding black wine to the string
black_wine_sample_str += f"\nblack wine sample: {black_wine_sample}"

# Write the formatted string to a text file
with open('black wine sample.txt', 'w', encoding='utf-8') as f:
    f.write(black_wine_sample_str)


# converting black wine sample to string()
black_wine_describe_str = black_wine_describe.to_string()

# Adding black wine to the string
black_wine_describe_str += f"\nblack wine describe: {black_wine_describe}"

# Write the formatted string to a text file
with open('black wine describe.txt', 'w', encoding='utf-8') as f:
    f.write(black_wine_describe_str)

# Exploring correlated columns
# Quality Column
sns.set(rc={"figure.figsize":(14,8)})
sns.countplot(x="quality", data=df_black)
plt.xlabel("quality")
plt.ylabel("Count")
plt.title("Count Plot Showing the Distribution of Black_Label_Wine Base on Quality", fontsize=14, fontweight='bold')
plt.show()

# Compute correlated columns
sns.pairplot(df_black)
plt.title("Scattered Plot Showing likely Combination of Columns in Black_Label_Dataset", fontsize=14, fontweight='bold')
plt.show()

#Compute heatmap
sns.heatmap(df_black.corr(),annot=True,fmt='.2f', linewidths=2)
plt.title("Heatmap Illustrating Correlation Between Columns in the Black_Label_Data", fontsize=14, fontweight='bold')
plt.show()

sns.distplot(df_black['alcohol level'])
plt.title("Distribution Graph Showing Alcohol Concentration with Respect to Quality of the Black_Label_Wine", fontsize=14, fontweight='bold')
plt.show()

# Variations in wine quality with respect to alcohol concentration

sns.boxplot(x='quality', y='alcohol level', data=df_black, showfliers=False)
plt.title("Boxplot Showing Variations in wine quality with respect to alcohol concentration in the Black_Label_Wine_Data", fontsize=14, fontweight='bold')
plt.show()

# correlation between alcohol and PH vslues
sns.jointplot(x='alcohol level', y='PH', data=df_black, kind='reg')
plt.title("Graph Showing correlation between alcohol and PH vslues in the Black_Label_Wine_Data", fontsize=14, fontweight='bold')
plt.show()

# Calculating correlation beween alcohol and PH value

def get_correlation(alcohol, PH, df_black):
    pearson_corr, p_value = pearsonr(df_black[alcohol], df_black[PH])
    print("Correlation between {} and {} is {}".format('alcohol level', 'PH', pearson_corr))
    print("P-value of this correlation is {}".format(p_value))

# Result
get_correlation('alcohol level', 'PH', df_black)
# computing the discriptive statistics

#Combining datasets (G_WINE_BLACK_LABEL_ANALYSIS)

print("black_wine_mean =", df_black["quality"].mean())
print("red_wine_mean =", df_red["quality"].mean())
# Adding new column_attribute
df_black["group"]="black Label"
df_red["group"]="red label"

# Compute for unique values

print('red_wine:list of "quality"', sorted(df_red['quality'].unique()))
print('black_wine:list of "quality"', sorted(df_black['quality'].unique()))

#creating  labels
low_value= "<=5"
medium="<=7"
High=">7"

# Apply the categorization based on the conditions
df_black['label'] = df_black['quality'].apply(lambda x: 'low' if x < 7 else ('medium' if x == 7 else 'high'))

# Apply the categorization based on the conditions
df_red['label'] = df_red['quality'].apply(lambda x: 'low' if x < 7 else ('medium' if x == 7 else 'high'))

# Printing the results
print(df_black["label"].value_counts())
print(df_red["label"].value_counts())

# Adding price column
def map_quality_to_price(label):
    if label=='low':
        return 10
    elif label == 'medium':
        return 15
    elif label == 'high':
        return 20
    else: 
         return None
df_black['price']=df_black['label'].apply(map_quality_to_price)
df_red['price']=df_red['label'].apply(map_quality_to_price)

# Joining the data using concatenation 

whisky_df= pd.concat([df_red, df_black])
print(whisky_df.head(100))
print(whisky_df["label"])

df_whiskies=pd.concat([df_black,df_red])

#Re-shuffle the rows to randomised.
df_whiskies=df_whiskies.sample(frac=1.0, random_state=42).reset_index(drop=True)
print(df_whiskies.head(50))


df_whiskies_str = df_whiskies.head(50).to_string()
df_whiskies_str += f"\ncombined dataset: {df_whiskies}"

#  formatted string to a text file
with open('rolls of combined dataset_results.txt', 'w', encoding='utf-8') as f:
    f.write(df_whiskies_str)
  
# Grouping columns based on attribute in the dataframe
subset_attribute_class=['alcohol level', 'PH', 'density level','quality']

low= round(df_whiskies[df_whiskies["label"]==  "low"][subset_attribute_class].describe(), 2)
medium = round(df_whiskies[df_whiskies['label']=="medium"][subset_attribute_class].describe(), 2)
high = round( df_whiskies[df_whiskies['label']== "high"][subset_attribute_class].describe(), 2)

# Joining the three class together

Grouped_whisky = pd.concat([low, medium, high ], axis=1, keys=  ['Low Quality Whisky', 'Medium Quality Whisky', 'High Quality Whisky'] )
print(Grouped_whisky)
# computing Univariate analysis

# Drawin Histogram
graph = df_whiskies.hist(bins=15, color= 'fuchsia', edgecolor= 'darkmagenta', linewidth=1.0, xlabelsize=10, ylabelsize=10, yrot=0, figsize=(8,6), grid=False)
plt.tight_layout(rect=(0,0,1.5,1.5))
plt.title("Varables and their distribution in the emerged dataset(Red_label + Black_label)")
plt.show()

# Multivariate computations
non_numeric_columns = df_whiskies.select_dtypes(exclude=['float64', 'int64']).columns
df_whiskies_numeric = df_whiskies.drop(columns=non_numeric_columns)

fig, (ax)= plt.subplots(1,1, figsize=(14,8))
Heatmap= sns.heatmap(df_whiskies_numeric.corr(), ax=ax, cmap='bwr', annot=True, fmt='.2f', linewidths=0.5)

# Defining subplot parameters
left = 0.1
bottom = 0.1
right = 0.9
top = 0.9
wspace = 0.2
hspace = 0.2

# Printing out the values
print("Left:", left)
print("Bottom:", bottom)
print("Right:", right)
print("Top:", top)
print("Wspace:", wspace)
print("Hspace:", hspace)

# Adjusting subplot parameters
fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

# Subplot with title
fig.suptitle('Joint whiskies and correlation Heatmap values', fontsize=14, fontweight='bold')

plt.show()

# Plotting countplot of discrete category using seaborn library

fig = plt.figure(figsize=(16,9))
sns.countplot(data=df_whiskies, x='quality', hue='group')

fig.suptitle('Group of Whiskies and their Frequency Distribution', fontsize=12, fontweight='bold')

plt.show()

fig = plt.figure(figsize=(16,9))
sns.countplot(data=df_whiskies, x='label', hue='group')
fig.suptitle(' label and their Frequency Distribution', fontsize=12, fontweight='bold')
plt.show()

fig = plt.figure(figsize=(16,9))
sns.countplot(data=df_whiskies, x='price', hue='group')

fig.suptitle('Price and their Frequency Distribution', fontsize=12, fontweight='bold')

plt.show()

# Creating axes
df_whiskies
fig = plt.figure(figsize=(16,12))
ax = fig.add_subplot(111, projection='3d')

# Adding columns

xscale = df_whiskies['sugar concentration']
yscale= df_whiskies['sulfur']
zscale= df_whiskies['sum of sulfur']
ax.scatter(xscale, yscale, zscale, s=50, alpha=0.6, edgecolors='w')

ax.set_xlabel('sugar concentration')
ax.set_ylabel('sulfur')
ax.set_zlabel('sum of sulfur')
ax.set_title('3-D Plot Showing Correlation Between Sugar Concentration, Sulfur & Sum of Sulfur', fontsize=12, fontweight='bold')
plt.show()

print(df_whiskies['label'].value_counts())

print(df_whiskies.columns)
label_counts = df_whiskies['label'].value_counts()
print(label_counts)

# Convert the DataFrame to a formatted string
label_counts_result_str = label_counts.to_string()

label_counts_result_str += f"\n label Counts: {label_counts}"

# Write the formatted string to a text file
with open('label_counts_result.txt', 'w', encoding='utf-8') as f:
    f.write(label_counts_result_str)
# Label encode the target variable ('label' column)
label_encoding = {'low': 1, 'medium': 2, 'high': 3}
df_whiskies['label'] = df_whiskies['label'].map(label_encoding)

# Encoding the ('group' column)
label_encoder = LabelEncoder()
df_whiskies['group_encoded'] = label_encoder.fit_transform(df_whiskies['group'])

# Dropping the original 'group' column
df_whiskies.drop('group', axis=1, inplace=True)

# Define features (X) and target variable (y) for classification
X_classification = df_whiskies.drop('alcohol level', axis=1)
y_classification = df_whiskies['label']

# Splitting the dataset for classification
X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(
    X_classification, y_classification, test_size=0.3, random_state=42)

# Initialize and train the logistic regression model for classification
logistic_model = LogisticRegression()
logistic_model.fit(X_train_classification, y_train_classification)

# Make predictions for classification
classification_predictions = logistic_model.predict(X_test_classification)

# Evaluate regression score & mean score
regression_score = logistic_model.score(X_test_classification, y_test_classification)

# Make predictions for Y values in regression
Y_values_predictions = logistic_model.predict(X_test_classification)
# Calculating Mean Square Error
mse = mean_squared_error(y_test_classification, Y_values_predictions)

# Create a DataFrame to compare actual vs. predicted values for regression
regression_evaluation = pd.DataFrame({'Actual': y_test_classification, 'Predicted': Y_values_predictions})
print("prediction_score/accuracy:", regression_score)
print(regression_evaluation.head(190))
print("Mean Squared Error:", mse)

# Convert the DataFrame to a formatted string
regression_result_str = regression_evaluation.head(190).to_string()

# Add the Mean Squared Error and Prediction Score/Accuracy to the string
regression_result_str += f"\nMean Squared Error: {mse}"
regression_result_str += f"\nPrediction Score/Accuracy: {regression_score}"

# Write the formatted string to a text file
with open('regression_evaluation_results.txt', 'w', encoding='utf-8') as f:
    f.write(regression_result_str)
