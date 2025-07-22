Reflective reasoning & insights throughout the project

### PROJECT AIM & CONTEXT
“I chose to apply KNN to the breast cancer diagnostic dataset from sklearn.
My aim was to see how a classical distance-based classifier could help predict whether a tumor is benign (1) or malignant (0), and build it as if it could run in production, not just in a notebook, but as modular code ready for deployment.”

🧪 OUTLIERS ANALYSIS SUMMARY
- The target variable in this dataset is binary:
  - 0 → malignant
  - 1 → benign
“I noticed that almost every feature had outliers.
Instead of removing them, I asked: why are they there?
By plotting the distribution of the target within these outliers, I discovered many outliers correspond to malignant cases.
So they aren’t noise, they’re critical edge cases a real model must learn from.
Rather than exclude them, I planned to control their influence using feature scaling and hyperparameter tuning.”

📊 CORRELATION ANALYSIS AND FEATURE SELECTION
“In the correlation matrix, I found strong correlations (some >0.9) among features measuring similar aspects of tumor size and shape, e.g., mean radius, mean perimeter, mean area.
I realized this could hurt KNN because highly correlated features double-count distance.
So I wrote a helper to keep only the most predictive feature in each highly correlated pair (i.e., the one with higher absolute correlation to the target).
This helps simplify the model, avoids redundancy, and improves performance.”

⚖ FEATURE SCALING
“KNN uses distance metrics, so feature scaling is critical.
After dropping highly correlated features, I scaled the remaining ones using StandardScaler.
This way, outliers won’t dominate the distance calculation, and each feature contributes proportionally.”

🧬 TARGET ENCODING
“The target was already numeric in sklearn: 1 means benign, 0 means malignant.
No further encoding was needed — that kept the pipeline simple and direct.”

⚙ FINAL DATASET & READINESS
“At this point, I had:
-Removed duplicate & missing checks (none were found)
-Visualized and analyzed outliers (kept them as informative)
-Reduced highly correlated features
-Scaled features
-Kept the target column numeric and ready
-The final dataset is clean, balanced, informative, and production-ready for train/test split and KNN model selection.”

 MODEL SELECTION & TRAINING
“Now that the data was cleaned, reduced, and scaled, I wanted to build the actual KNN classifier.
Instead of just picking a k by intuition like an experienced ML Engineer would accurately do, which i am not, yet :), I used cross-validation to let the data tell me the best k.
Why? Because KNN is very sensitive to k:
-Low k can overfit (too sensitive to noise)
-High k can underfit (too smooth, ignores details).
-Cross-validation on the training set helps pick the k that generalizes best, instead of fitting only to one lucky split.”

🔧 USING DISTANCE WEIGHTS
“I set weights='distance' in KNeighborsClassifier.
Why? Because in real-world medical data, closer cases should have more influence when predicting an unknown case.
Uniform weighting would treat all neighbors equally, even those far away.
Using distance weighting helps the model stay sensitive to strong local signals.”
"Learned this from ChatGPT and DeepSeek, I use them for study too, apart from books and prerecorded classes"

✂️ STRATIFIED SPLIT
“When splitting the data into train and test, I used stratification.
Why? Because the classes aren’t perfectly balanced (more benign than malignant).
Stratified split keeps the same class proportions in both train and test sets — critical in healthcare, so test results reflect real-world prevalence.”

📊 PLOTTING K VS ACCURACY
“After tuning k, I plotted k vs accuracy.
-This is more than just a diagnostic:
-It visually shows if performance is stable or sensitive to k.
-Helps justify why I picked the chosen k in reports to stakeholders.”

⚙ FINAL MODEL
“With the best k from cross-validation, I trained the final KNN model on the training data.
Then I evaluated it on the test set, capturing accuracy, F1 score, ROC AUC, and the full classification report.
Why F1 and ROC AUC? Because in cancer diagnosis, catching malignant cases matters more than raw accuracy.
F1 balances precision & recall, and ROC AUC shows discrimination ability.”
"Also read this from the LLMs"

💾 SAVING ARTIFACTS
“Finally, I saved the trained model with joblib, the k vs accuracy plot, and the metrics report as markdown.
Why? To make the workflow reproducible, deployable, and explainable.
This means someone else (or me later) can load the model and run predictions without retraining.”

“I wrapped everything in functions, and used:
if __name__ == "__main__":
    main()
so the script only runs when executed directly. Something i just learned recently
This makes it both:
A script you can run (python train.py)
A module you can import from elsewhere to reuse functions.”

“Instead of treating this like a quick notebook demo, I built it as if it had to go into production:
modular, explainable, reproducible, and tested.
The choices (stratified split, cross-validation, distance weights, scaling, pruning correlated features) all came from thinking about what makes KNN work in real life — and what could hurt it if ignored.”
"Eager to keep learning and improving as a Data Scientist and ML Engineer, I really mean business."

