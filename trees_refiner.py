###Import Recipe to be Optimized and ran through Refiner Code
recipe_for_refiner = 'SINGLE ROW RECIPE TO REFINE HERE.csv'
training_data = 'TRANING DATA HERE.csv'

# Imports
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

#Connect to Google Drive
from google.colab import drive
drive.mount('/content/drive')

#Load the data sets (pre)
train_df = pd.read_csv(training_data)
test_df = pd.read_csv(recipe_for_refiner)

##Preprocessing, cleaning, and imputation of training data

#Remove the row label column in the training and testing data sets
train_df.drop(train_df.columns[0], axis=1, inplace=True)
test_copy = test_df.copy()
test_df.drop(test_df.columns[0], axis=1, inplace=True)


#Drop target column, and assign target column
y_train = train_df['Customer Score']
test_df.pop('Customer Score')
train_df.pop('Customer Score')

# Standardization of training data
scaler = StandardScaler()
scaler.fit(train_df)
train_df = scaler.transform(train_df)

#Standardization of the test data & Model pipeline
pipeline = Pipeline([
    ('scaler', scaler),
    ('model', ExtraTreesRegressor(random_state=42)) #Remove 'random state' if you do not want it to be consistent scores
])

# Hyperparameters
hyperparams = {
  'model__n_estimators': [100, 300, 500, 600, 700],
  'model__max_depth': [None, 5, 8, 10, 15, 20],
  'model__min_samples_split': [2, 5, 7, 15, 25, 50]
}

# Grid search
gridsearch = GridSearchCV(pipeline, hyperparams, cv=5, n_jobs=-1)
gridsearch.fit(train_df, y_train)

# Best model
best_model = gridsearch.best_estimator_
print(best_model)

#Now Average the two together, and place them into the Final dataframe for review
X_test_scaled = StandardScaler().fit_transform(test_df)
y_pred = best_model.predict(X_test_scaled)
test_copy["Customer Score"] = y_pred

##Begin Refiner Code

##Mash temp first

#Select top rated recipe in Final DF
top_recipe_row = test_copy[test_copy['Customer Score'] == test_copy['Customer Score'].max()]
top_recipe_row = top_recipe_row.sample(n=1)
print(top_recipe_row)

#Duplicate the top recipe 12 times for mash temp (146 - 158'F)
mashtemp_df = pd.concat([top_recipe_row] * 12, ignore_index=True)
mashtemp_df['Mash Temperature (Sacch Step)'] = list(range(146, 158))

#Copy the original mashtemp df (for final score). Scale the test data / target column data
mashtemp_df_copy = mashtemp_df.copy()
mashtemp_df.pop('Customer Score')
mashtemp_df.pop('Beer Name')
mashtemp_df_scaled = StandardScaler().fit_transform(mashtemp_df)

#Run the same model from above on the new mashtemp scaled df.  Attach the scores to the unscaled and copied data.
mashtemp_pred = best_model.predict(mashtemp_df_scaled)
mashtemp_df_copy['Customer Score'] = mashtemp_pred

#Now for Yeast Strains

#Select the top scored recipe for mashtemp df
top_mashtemp_row = mashtemp_df_copy[mashtemp_df_copy['Customer Score'] == mashtemp_df_copy['Customer Score'].max()]
top_mashtemp_row = top_mashtemp_row.sample(n=1)
print(top_mashtemp_row)

#Select amount of columns at and in between BSI & Jasper columns. Duplicate the top rated mash temp row by this amount, and make
#a new dataframe
start_idx = top_mashtemp_row.columns.get_loc('BSI Barbarian')
end_idx = top_mashtemp_row.columns.get_loc('Jasper Voss')
num_columns = end_idx - start_idx + 1
yeast_df = pd.concat([top_mashtemp_row] * num_columns, ignore_index=True)

for i, col in enumerate(yeast_df.columns[start_idx:end_idx+1]):
    yeast_df[col] = 0
    yeast_df.at[i, col] = 1

#Adjust the temp columns accordingly
# Get column names that match the filter criteria
kviek_cols = yeast_df.filter(like="Kviek Strain Primary Ferm").columns
lager_cols = yeast_df.filter(like="Lager Temps").columns
ale_cols = yeast_df.filter(like="Ale Strain Primary Ferm").columns

# Combine the column names
ale_cols_to_update = kviek_cols.union(lager_cols)
kviek_cols_to_update = ale_cols.union(lager_cols)
lager_cols_to_update = kviek_cols.union(ale_cols)

# Update the values for ale strains
# Set 'temp1' to '1' where ale strains are '1'
yeast_df.loc[yeast_df['BSI Barbarian'] == 1, ale_cols] = 1
yeast_df.loc[yeast_df['Thiolized Hazy'] == 1, ale_cols] = 1
yeast_df.loc[yeast_df['Melon Drop'] == 1, ale_cols] = 1
yeast_df.loc[yeast_df['Imperial AO-4'] == 1, ale_cols] = 1
yeast_df.loc[yeast_df['Jasper Hefeweizen'] == 1, ale_cols] = 1
yeast_df.loc[yeast_df['Jasper Conan'] == 1, ale_cols] = 1
yeast_df.loc[yeast_df['Jasper Saison'] == 1, ale_cols] = 1
yeast_df.loc[yeast_df['K97 Kolsch'] == 1, ale_cols] = 1
yeast_df.loc[yeast_df['S-04'] == 1, ale_cols] = 1
yeast_df.loc[yeast_df['White Labs T. Delbreuckii'] == 1, ale_cols] = 1
#yeast_df.loc[yeast_df['London Ale III'] == 1, ale_cols] = 1
yeast_df.loc[yeast_df['US-05'] == 1, ale_cols] = 1
yeast_df.loc[yeast_df['ME Sour Dough Isolate'] == 1, ale_cols] = 1
yeast_df.loc[yeast_df['Maniacal Wickerhamomyces'] == 1, ale_cols] = 1

#Set all other temps to 0
# Set 'temp1' to '1' where ale strains are '1'
yeast_df.loc[yeast_df['BSI Barbarian'] == 1, ale_cols_to_update] = 0
yeast_df.loc[yeast_df['Thiolized Hazy'] == 1, ale_cols_to_update] = 0
yeast_df.loc[yeast_df['Melon Drop'] == 1, ale_cols_to_update] = 0
yeast_df.loc[yeast_df['Imperial AO-4'] == 1, ale_cols_to_update] = 0
yeast_df.loc[yeast_df['Jasper Hefeweizen'] == 1, ale_cols_to_update] = 0
yeast_df.loc[yeast_df['Jasper Conan'] == 1, ale_cols_to_update] = 0
yeast_df.loc[yeast_df['Jasper Saison'] == 1, ale_cols_to_update] = 0
yeast_df.loc[yeast_df['K97 Kolsch'] == 1, ale_cols_to_update] = 0
yeast_df.loc[yeast_df['S-04'] == 1, ale_cols_to_update] = 0
yeast_df.loc[yeast_df['White Labs T. Delbreuckii'] == 1, ale_cols_to_update] = 0
#yeast_df.loc[yeast_df['London Ale III'] == 1, ale_cols_to_update] = 0
yeast_df.loc[yeast_df['US-05'] == 1, ale_cols_to_update] = 0
yeast_df.loc[yeast_df['ME Sour Dough Isolate'] == 1, ale_cols_to_update] = 0
yeast_df.loc[yeast_df['Maniacal Wickerhamomyces'] == 1, ale_cols_to_update] = 0

#Adjust columns for lager strains
yeast_df.loc[yeast_df['Jasper Bo Lager'] == 1, lager_cols_to_update] = 0
yeast_df.loc[yeast_df['Jasper Augustiner'] == 1, lager_cols_to_update] = 0
yeast_df.loc[yeast_df['Jasper Mexican Lager'] == 1, lager_cols_to_update] = 0
yeast_df.loc[yeast_df['Czech Lager'] == 1, lager_cols_to_update] = 0

#Adjust columns for kviek strains
yeast_df.loc[yeast_df['Omega Hothead'] == 1, kviek_cols_to_update] = 0
yeast_df.loc[yeast_df['Omega Hornindal'] == 1, kviek_cols_to_update] = 0
yeast_df.loc[yeast_df['Jasper Voss'] == 1, kviek_cols_to_update] = 0

#Copy the original yeast df (for final score). Scale the test data / target column data
yeast_df_copy = yeast_df.copy()
yeast_df.pop('Customer Score')
yeast_df.pop('Beer Name')
yeast_df_scaled = StandardScaler().fit_transform(yeast_df)

#Run the same model from above on the new yeast scaled df.  Attach the scores to the unscaled and copied data.
yeast_pred = best_model.predict(yeast_df_scaled)
yeast_df_copy['Customer Score'] = yeast_pred

#Now for Water Salts

#Cacl first

#Select the top scored recipe for yeast df
top_yeast_row = yeast_df_copy[yeast_df_copy['Customer Score'] == yeast_df_copy['Customer Score'].max()]
top_yeast_row = top_yeast_row.sample(n=1)
print(top_yeast_row)

#Duplicate the top recipe 3000 times for CaCl
cacl_df = pd.concat([top_yeast_row] * 3000, ignore_index=True)
cacl_df['Calcium Chloride (Grams)'] = list(range(0, 3000))
cacl_df['Total Water Salts'] = cacl_df['Calcium Chloride (Grams)'] + cacl_df['Gypsum (Grams)'] + cacl_df['Potassium Chloride (Grams)']
cacl_df['Potassium Chloride ratio'] = cacl_df['Potassium Chloride (Grams)'] / cacl_df['Total Water Salts'] * 10
cacl_df['Calcium Chloride Ratio'] = cacl_df['Calcium Chloride (Grams)'] / cacl_df['Total Water Salts'] * 10
cacl_df['Gypsum Ratio'] = cacl_df['Gypsum (Grams)'] / cacl_df['Total Water Salts'] * 10

#copy the original cacl df (for final score). Scale the test data / target column data
cacl_df_copy = cacl_df.copy()
cacl_df.pop('Customer Score')
cacl_df.pop('Beer Name')
cacl_df_scaled = StandardScaler().fit_transform(cacl_df)

#Run the same model from above on the new cacl scaled df.  Attach the scores to the unscaled and copied data.
cacl_pred = best_model.predict(cacl_df_scaled)
cacl_df_copy['Customer Score'] = cacl_pred

#Now for Gypsum

#Select the top scored recipe for Cacl df
top_cacl_row = cacl_df_copy[cacl_df_copy['Customer Score'] == cacl_df_copy['Customer Score'].max()]
top_cacl_row = top_cacl_row.sample(n=1)
print(top_cacl_row)

#Duplicate the top recipe 3000 times for gypsum
gypsum_df = pd.concat([top_cacl_row] * 3000, ignore_index=True)
gypsum_df['Gypsum (Grams)'] = list(range(0, 3000))
gypsum_df['Total Water Salts'] = gypsum_df['Calcium Chloride (Grams)'] + gypsum_df['Gypsum (Grams)'] + gypsum_df['Potassium Chloride (Grams)']
gypsum_df['Potassium Chloride ratio'] = gypsum_df['Potassium Chloride (Grams)'] / gypsum_df['Total Water Salts'] * 10
gypsum_df['Calcium Chloride Ratio'] = gypsum_df['Calcium Chloride (Grams)'] / gypsum_df['Total Water Salts'] * 10
gypsum_df['Gypsum Ratio'] = gypsum_df['Gypsum (Grams)'] / gypsum_df['Total Water Salts'] * 10

#copy the original gypsum df (for final score). Scale the test data / target column data
gypsum_df_copy = gypsum_df.copy()
gypsum_df.pop('Customer Score')
gypsum_df.pop('Beer Name')
gypsum_df_scaled = StandardScaler().fit_transform(gypsum_df)

#Run the same model from above on the new gypsum scaled df.  Attach the scores to the unscaled and copied data.
gypsum_pred = best_model.predict(gypsum_df_scaled)
gypsum_df_copy['Customer Score'] = gypsum_pred

#Now for KCl

#Select the top scored recipe for gypsum df
top_gypsum_row = gypsum_df_copy[gypsum_df_copy['Customer Score'] == gypsum_df_copy['Customer Score'].max()]
top_gypsum_row = top_gypsum_row.sample(n=1)
print(top_gypsum_row)

#Duplicate the top recipe 3000 times for KCl
kcl_df = pd.concat([top_gypsum_row] * 3000, ignore_index=True)
kcl_df['Potassium Chloride (Grams)'] = list(range(0, 3000))
kcl_df['Total Water Salts'] = kcl_df['Calcium Chloride (Grams)'] + kcl_df['Gypsum (Grams)'] + kcl_df['Potassium Chloride (Grams)']
kcl_df['Potassium Chloride ratio'] = kcl_df['Potassium Chloride (Grams)'] / kcl_df['Total Water Salts'] * 10
kcl_df['Calcium Chloride Ratio'] = kcl_df['Calcium Chloride (Grams)'] / kcl_df['Total Water Salts'] * 10
kcl_df['Gypsum Ratio'] = kcl_df['Gypsum (Grams)'] / kcl_df['Total Water Salts'] * 10

#copy the original kcl df (for final score). Scale the test data / target column data
kcl_df_copy = kcl_df.copy()
kcl_df.pop('Customer Score')
kcl_df.pop('Beer Name')
kcl_df_scaled = StandardScaler().fit_transform(kcl_df)

#Run the same model from above on the new kcl scaled df.  Attach the scores to the unscaled and copied data.
kcl_pred = best_model.predict(kcl_df_scaled)
kcl_df_copy['Customer Score'] = kcl_pred

###Now to do the hops

#WP Hops first

#Select the top scored recipe for kcl df
top_kcl_row = kcl_df_copy[kcl_df_copy['Customer Score'] == kcl_df_copy['Customer Score'].max()]
top_kcl_row = top_kcl_row.sample(n=1)
print(top_kcl_row)

#Duplicate the top recipe 60 times for WP Hops (60 lbs could be considered the limit)
wp_df = pd.concat([top_kcl_row] * 60, ignore_index=True)
wp_df['Total WP Hops'] = list(range(0, 60))
wp_df['Total Hops Used'] = wp_df['Total WP Hops'] + wp_df['Total Hops Dry Hops']
wp_df['Ratio: Dry Hops'] = wp_df['Total Hops Dry Hops'] / wp_df['Total Hops Used']
wp_df['Ratio: WP hops'] = wp_df['Total WP Hops'] / wp_df['Total Hops Used']

#copy the original wp df (for final score). Scale the test data / target column data
wp_hops_copy = wp_df.copy()
wp_df.pop('Customer Score')
wp_df.pop('Beer Name')
wp_hops_scaled = StandardScaler().fit_transform(wp_df)

#Run the same model from above on the new wp_hops scaled df.  Attach the scores to the unscaled and copied data.
wp_hops_pred = best_model.predict(wp_hops_scaled)
wp_hops_copy['Customer Score'] = wp_hops_pred

#Now for Dry Hops

#Select the top scored recipe for wp_hops df
top_wp_hops_row = wp_hops_copy[wp_hops_copy['Customer Score'] == wp_hops_copy['Customer Score'].max()]
top_wp_hops_row = top_wp_hops_row.sample(n=1)
print(top_wp_hops_row)

#Duplicate the top recipe 180 times for Dry Hops
dryhop_df = pd.concat([top_wp_hops_row] * 180, ignore_index=True)
dryhop_df['Total Hops Dry Hops'] = list(range(0, 180))
dryhop_df['Total Hops Used'] = dryhop_df['Total Hops Dry Hops'] + dryhop_df['Total WP Hops']
dryhop_df['Ratio: Dry Hops'] = dryhop_df['Total Hops Dry Hops'] / dryhop_df['Total Hops Used']
dryhop_df['Ratio: WP hops'] = dryhop_df['Total WP Hops'] / dryhop_df['Total Hops Used']

#copy the original dryhop df (for final score). Scale the test data / target column data
dryhop_df_copy = dryhop_df.copy()
dryhop_df.pop('Customer Score')
dryhop_df.pop('Beer Name')
dryhop_df_scaled = StandardScaler().fit_transform(dryhop_df)

#Run the same model from above on the new dryhop scaled df.  Attach the scores to the unscaled and copied data.
dryhop_pred = best_model.predict(dryhop_df_scaled)
dryhop_df_copy['Customer Score'] = dryhop_pred

#Select the top dry hop row
top_dryhop_row = dryhop_df_copy[dryhop_df_copy['Customer Score'] == dryhop_df_copy['Customer Score'].max()]
top_dryhop_row = top_dryhop_row.sample(n=1)
print(top_dryhop_row)

########Refiner Completed. Check Drive ->>
top_dryhop_row.to_csv('WRITE REFINED RECIPE NAME HERE.csv',index=False)
