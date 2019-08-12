# 1. Preprocessing
- preprocess train, test, structures dataframes
- create an object holding preprocessed dataset

# 2. Sub-Target Selection
- select which target is predicted except "scalar_coupling_constant"
- The sub-target candidates are "fc", "sd", "pso", and "dso"

# 3. Creating Feature
- create features for learning model
- not include the results predicted as sub-targets

# 4. Learning Model
- learn model with crated features
- need parameter tuning on each prediction

# 5. Target Prediction
- predict the selected target with the learned model

# 6. Creating Final Feature
- come to this phase after predicting all sub-targets
- create final features for learning final model
- include the results predicted as sub-targets

# 7. Learning Final Model
- learn final model with the created final features

# 8. Final Target Prediction
- predict the final target with the final learned model

# 9. Submission
- make the submission file
