BEER RECIPE REFINER

Provide a single row (or recipe) to the code, and training data with actual customer scores.  The code will then re-adjust the numbers and volumes of ingredients to amplify the predicted customer score.

This code also includes imputation for missing data for specific columns.

Make sure all columns called for in the code are present in both the training set and the single recipe set for refinement.

This uses the Extra Trees algorithm, as it seemed to have always performed the best, however, feel free to use whatever algorithm you'd like that uses regression.
