# Machine Learning–Based Health Insurance Premium Prediction System 

<p align="left">
  <img src="https://img.shields.io/badge/Made%20With-Colab-blue?logo=googlecolab&logoColor=white" alt="Made with Colab">
  <img src="https://img.shields.io/badge/Language-Python-green?logo=python" alt="Language: Python">
  <img src="https://img.shields.io/badge/💻Dev%20Environment-VS%20Code-blue?logo=visualstudiocode">
  <img src="https://img.shields.io/badge/⚖️%20License-MIT-red" alt="License">
  <img src="https://img.shields.io/badge/🐞%20Issues-None-green" alt="Issues">
  <img src="https://img.shields.io/github/repo-size/ShaikhBorhanUddin/Machine-Learning-Based-Health-Insurance-Premium-Prediction-System?logo=github" />
  <img src="https://img.shields.io/github/last-commit/ShaikhBorhanUddin/Machine-Learning-Based-Health-Insurance-Premium-Prediction-System" alt="Last Commit">
  <img src="https://img.shields.io/badge/🤖%20Models-Random%20Forest | XGBoost | ElasticNet | LightGBM-red" alt="Models">
  <img src="https://img.shields.io/badge/🗂️Dataset-Kaggle-blueviolet" alt="Dataset: Health Insurance">
  <img src="https://img.shields.io/badge/⚙️Runtime-CPU-blue" alt="Runtime"> 
  <img src="https://img.shields.io/badge/📊%20Visualization-Matplotlib | Seaborn | SHAP-yellow" alt="Visualization & Explainability">
  <img src="https://img.shields.io/badge/🎨%20UI-Streamlit | HTML | CSS-purple" alt="UI Stack">
  <img src="https://img.shields.io/badge/Deployment-Streamlit-orange?logo=streamlit" alt="Deployment: Streamlit">
  <img src="https://img.shields.io/badge/Version%20Control-Git-orange?logo=git" alt="Git">
  <img src="https://img.shields.io/badge/Host-GitHub-green?logo=github" alt="GitHub">
  <img src="https://img.shields.io/github/forks/ShaikhBorhanUddin/Machine-Learning-Based-Health-Insurance-Premium-Prediction-System?style=social" alt="Forks">
  <img src="https://img.shields.io/badge/🏁Project-Deployed-brightgreen" alt="Status">
</p> 

![Dashboard](Assets/title_image_new.png) 

## Overview 

## Folder Structure 

All code, Python notebooks, trained models, and images used in this project are organized and stored in this repository according to the following folder structure: 

```bash
Health Insurance Premium Prediction Project
│
├── Assets/                                   # Images for project documentation
├── Dataset/                              
│      ├── medical_insurance.csv      
│      ├── medical-insurance_cleaned.csv                   
│      └── medical_insurance_cleaned_engineered.csv 
├── Models/
│      ├── lightgbm_model.pkl                       
│      ├── xgboost_model_cpu.pkl              # To ensure model is trained on cpu instead of gpu
│      ├── xgboost_model.pkl
│      ├── elasticnet_model.pkl
│      └── random_forest_model.pkl            # Excluded from repository due to large size (674 MB)
├── Notebooks/                                
│      ├── data_cleaning.ipynb
│      ├── feature_engineering.ipynb
│      ├── EDA.ipynb
│      ├── train_test.ipynb
│      ├── xgboost_train_test.ipynb
│      └── xgboost_SHAP.ipynb
├── app.py                                    # Deployment code
├── requirements.txt                          # Python dependencies for deployment
├── README.md                                 # Project documentation
└── Licence.md
```

## Project Workflow 

The following diagram illustrates the end-to-end pipeline of the project, starting from raw data ingestion and preprocessing to feature engineering, model training, evaluation, and final deployment. 

![Dashboard](Assets/workflow_corrected.png) 

## Dataset 

The original [Raw Dataset](https://www.kaggle.com/datasets/mohankrishnathalla/medical-insurance-cost-prediction) contains more than 50 features, including identifiers, claim aggregates, and variables that may introduce potential data leakage. For clearer analysis and modeling, related variables are organized into logical categories such as demographics, lifestyle factors, medical conditions, healthcare utilization, procedures, and insurance policy attributes, making the dataset easier to interpret and analyze. The dataset consists of 100,000 records with 54 features, capturing demographic, socioeconomic, lifestyle, clinical, and insurance-related information. With a size of approximately 21 MB in CSV format, the dataset is relatively lightweight and well-suited for experimentation in typical machine learning environments. Most variables are fully populated, with missing values present only in the `alcohol_freq` column, which were handled during preprocessing. Overall, the dataset includes a combination of 44 numerical features and 10 categorical features, providing a diverse set of predictors for exploring patterns related to healthcare utilization and insurance premium estimation. 

| Feature Group            | Column Names                                                                                                                                              | Category              | Data Type               | Description                                                                                              |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- | ----------------------- | -------------------------------------------------------------------------------------------------------- |
| Identifier               | `person_id`                                                                                                                                               | Identifier            | Integer                 | Unique identifier for each individual record in the dataset.                                             |
| Demographics             | `age`, `sex`, `region`, `urban_rural`                                                                                                                     | Demographic           | Numerical / Categorical | Basic personal characteristics describing the individual's location and gender.                          |
| Socioeconomic            | `income`, `education`, `marital_status`, `employment_status`                                                                                              | Socioeconomic         | Numerical / Categorical | Socioeconomic background variables influencing healthcare access and insurance risk.                     |
| Household Information    | `household_size`, `dependents`                                                                                                                            | Demographic           | Integer                 | Indicates number of household members and financial dependents.                                          |
| Lifestyle Factors        | `bmi`, `smoker`, `alcohol_freq`                                                                                                                           | Health / Lifestyle    | Numerical / Categorical | Behavioral and health indicators associated with medical risk factors.                                   |
| Healthcare Utilization   | `visits_last_year`, `hospitalizations_last_3yrs`, `days_hospitalized_last_3yrs`, `medication_count`                                                       | Medical Utilization   | Integer                 | Measures the individual's healthcare usage frequency and hospitalization history.                        |
| Vital Health Indicators  | `systolic_bp`, `diastolic_bp`, `ldl`, `hba1c`                                                                                                             | Clinical Measurements | Float                   | Medical test results indicating cardiovascular and metabolic health status.                              |
| Insurance Policy Details | `plan_type`, `network_tier`, `deductible`, `copay`, `policy_term_years`, `policy_changes_last_2yrs`                                                       | Insurance             | Categorical / Numerical | Characteristics of the individual's health insurance policy and coverage structure.                      |
| Risk & Provider Metrics  | `provider_quality`, `risk_score`                                                                                                                          | Derived Feature       | Float                   | Risk assessment indicators derived from health profile and provider quality rating.                      |
| Cost & Premium Variables | `annual_medical_cost`, `annual_premium`, `monthly_premium`                                                                                                | Financial             | Float                   | Total medical spending and insurance premium costs associated with the policy.                           |
| Claims Features          | `claims_count`, `avg_claim_amount`, `total_claims_paid`                                                                                                   | Insurance Claims      | Numerical               | Historical insurance claim statistics reflecting claim frequency and claim payment amounts.              |
| Chronic Condition Count  | `chronic_count`                                                                                                                                           | Medical               | Integer                 | Total number of chronic conditions recorded for the individual.                                          |
| Medical Conditions       | `hypertension`, `diabetes`, `asthma`, `copd`, `cardiovascular_disease`, `cancer_history`, `kidney_disease`, `liver_disease`, `arthritis`, `mental_health` | Medical               | Binary                  | Presence or absence of major chronic or health conditions affecting risk profile.                        |
| Medical Procedures       | `proc_imaging_count`, `proc_surgery_count`, `proc_physio_count`, `proc_consult_count`, `proc_lab_count`                                                   | Medical Utilization   | Integer                 | Counts of different medical procedures performed for the individual.                                     |
| Risk Label               | `is_high_risk`                                                                                                                                            | Target / Derived      | Binary                  | Indicates whether the individual is classified as high risk (potential data leakage if used as feature). |
| Major Procedure Flag     | `had_major_procedure`                                                                                                                                     | Medical               | Binary                  | Indicates whether the individual has undergone a major medical procedure.                                | 

The detailed data cleaning and feature engineering processes are discussed in the following sections. 

## Data Cleaning 

During the data cleaning stage, several columns were removed from the raw dataset to improve model reliability and ensure that the final features are suitable for real-world prediction scenarios. The column `person_id` was dropped because it is simply a unique identifier and does not contribute meaningful predictive information. Similarly, `provider_quality` and `risk_score` were excluded because these variables would not be available as inputs from end users of the application. In addition, the `risk_score` variable could introduce data leakage since it may already incorporate information related to the target outcome. The `monthly_premium` column was also removed because annual_premium is used as the target variable, making the monthly value redundant.

Several additional variables were removed because they represent historical insurance or claims information that application users would not realistically be able to provide. These include `policy_changes_last_2yrs`, `claims_count`, `avg_claim_amount`, and `total_claims_paid`. The column `chronic_count` was also dropped because it is simply an aggregate of ten individual disease indicator columns, making it redundant. Furthermore, the procedure-related variables `proc_imaging_count`, `proc_surgery_count`, `proc_physio_count`, `proc_consult_count`, and `proc_lab_count` were removed because they are vague and may represent medical procedures that are not appropriate or practical for machine learning prediction in this context. Finally, `is_high_risk` was excluded due to the potential for data leakage, as it likely reflects information closely related to the target variable. 

The alcohol_freq column contains the unique values 

```bash
[NaN, 'Weekly', 'Daily', 'Occasional']
```

A large portion of the entries (30,083 rows) had missing values. Since alcohol consumption is typically reported when applicable, it was reasonably assumed that these missing values represent individuals who do not consume alcohol. Therefore, the null values were imputed with the value `Never` to indicate no alcohol consumption. 

The education column contains the unique values 

```bash
['Doctorate', 'No HS', 'HS', 'Some College', 'Masters','Bachelors']
```

Some of these categories ('No HS', 'HS', and 'Some College') were abbreviated and somewhat unclear in meaning. To improve clarity and interpretability, these values were remapped to more descriptive labels, while the remaining categories were kept unchanged. This mapping enhances the readability and consistency of the education-level information in the dataset. 

```bash
{'No HS': 'High School Dropout', 'HS': 'High School', 'Some College': 'College'}
```

Finally, the entries in the trem_type column were replaced with more descriptive labels. 

```bash
{'PPO': 'Preferred Provider Organization', 'POS': 'Point-of-Service', 'HMO': 'Health Maintenance Organization', 'EPO': 'Exclusive Provider Organization'}
``` 

This transformation was performed to improve the clarity and interpretability of the variable, ensuring that the category names are easier to understand during analysis and model interpretation. With these adjustments, the data cleaning stage is complete, and the dataset is now prepared for the subsequent feature engineering process. 

## Feature Engineering 

The feature engineering process begins with the blood pressure–related variables. While the `systolic_bp` and `diastolic_bp` columns are suitable for machine learning computations, end users are generally more familiar with descriptive medical terms such as ***Low Blood Pressure*** or ***Hypertension***. To improve interpretability, a new feature named `bp_category` was created to categorize blood pressure readings into clinically meaningful groups, according to guidelines provided by the American Heart Association (AHA). 

| bp_category          | systolic_bp     | diastolic_bp     |
| -------------------- | --------------- | ---------------- |
| Low Blood Pressure   | < 90            | < 60             |
| Normal               | 90–129          | 60–80            |
| Elevated             | 120–129         | < 80             |
| Hypertension Stage 1 | 130–139         | 80–89            |
| Hypertension Stage 2 | ≥ 140           | ≥ 90             | 

A new categorical feature `bmi_group` was also created from the continuous `bmi` variable. Although BMI values are useful for numerical modeling, grouping them into clinically recognized categories improves interpretability during exploratory analysis, model explanation and deployment UI. The categorization follows the widely used World Health Organization (WHO) BMI classification standards. 

| bmi_group    | BMI Range   |
| ------------ | ----------- |
| Underweight  | < 18.5      |
| Normal       | 18.5 – 24.9 |
| Overweight   | 25 – 29.9   |
| Obese I      | 30 – 34.9   |
| Obese II     | 35 – 39.9   |
| Obese III    | ≥ 40        | 

An additional categorical variable `ldl_group` was created from the continuous `ldl` cholesterol measurements. While the original numeric values are informative for machine learning models, grouping them into clinically recognized ranges enhances interpretability and aligns the dataset with standard medical guidelines. 

| ldl_group       | LDL Range (mg/dL) |
| --------------- | ----------------- |
| Optimal         | < 100             |
| Near Optimal    | 100 – 129         |
| Borderline High | 130 – 159         |
| High            | 160 – 189         |
| Very High       | ≥ 190             | 

Similarly, `hba1c_group` was created from the continuous `HbA1c` measurements. While the numeric values are important for modeling, grouping them into clinically meaningful categories (such as normal, prediabetes, and diabetes) makes the data more interpretable for users who are familiar with these common clinical terms rather than the raw numeric scale. 

| hba1c_group | HbA1c (%) |
| ----------- | --------- |
| Normal      | < 5.7     |
| Prediabetes | 5.7 – 6.4 |
| Diabetes    | ≥ 6.5     | 

finally, `annual_medical_cost_grouped` was created from the continuous `annual_premium` values. While the numeric premium amounts are useful for modeling, grouping them into meaningful categories improves interpretability and helps users quickly understand cost levels. 

| annual_medical_cost_grouped| Annual Premium Range ($) |
| -------------------------- | ------------------------ |
| Very Low                   | 0 – 499                  |
| Low                        | 500 – 1,999              |
| Medium                     | 2,000 – 4,999            |
| High                       | 5,000 – 9,999            |
| Very High                  | 10,000+                  | 

With the creation of these five derived categorical features, the feature engineering process is complete. Although they are not used in machine learning training due to redundancy with the original numeric variables, they are retained for deployment to support automatically generated fields, enhancing the user interface and overall interpretability. 

## Exploratory Data Analysis 

To ensure meaningful analysis, the exploratory data analysis was conducted using cleaned and feature-engineered dataset (derived from the raw dataset). This allows the analysis to focus only on relevant demographic, socioeconomic, and risk-related features that are actually used by the machine learning models. 

![Dashboard](Assets/EDA_demographics.png) 

The EDA begins with an examination of the demographic characteristics of insurance holders in the dataset. The age distribution shows that the majority of individuals fall within the range of 31 to 60 (the 46–60 group being the largest), indicating that middle-aged individuals dominate the dataset. Gender distribution is nearly balanced, with female (≈49.2%) and male (≈48.8%) populations being almost equal, while a very small fraction falls under the "Other" category. Household size analysis reveals that smaller households (homes with 2–3 rooms) are more common, while larger households (6 or more rooms) appear much less frequently. In terms of education, individuals with Bachelor’s degree holders represent the largest share (≈28%), followed by college and high school education, indicating that the dataset primarily consists of individuals with moderate to higher levels of education. 

![Dashboard](Assets/EDA_location.png) 

Regional distribution shows that the South region has the highest representation, followed by the North and East, while the Central region contributes the smallest share. In terms of residential classification, the majority of individuals reside in urban areas (≈60%), with smaller proportions living in suburban (≈25%) and rural areas (≈15%), suggesting that the dataset is somewhat urban-centric. Additionally, the dependents analysis shows that most individuals have 0–1 dependents, with the frequency decreasing significantly as the number of dependents increases. 

![Dashboard](Assets/EDA_financials.png) 

Over half of the individuals are married (≈53%), while single individuals account for about 36%, and smaller proportions are divorced or widowed. Employment status shows that most individuals are employed (≈55%), with the remaining population distributed among retired, unemployed, and self-employed groups. Income distribution indicates that a large portion of individuals fall within the lower to middle income brackets, particularly below $50K, while relatively fewer individuals belong to higher income groups above $100K. 

<p align="center">
  <img src="Assets/EDA_habits.png" width="40.2%" />
  <img src="Assets/EDA_diseases.png" width="58.8%" />
</p> 

Smoking behavior shows that the majority of individuals are non-smokers, while smaller proportions are former or current smokers. Alcohol consumption patterns indicate that occasional drinking is the most common, followed by individuals who never consume alcohol, with relatively fewer reporting weekly or daily consumption. The LDL cholesterol distribution reveals that most individuals fall within the near-optimal and optimal ranges, although a noticeable portion still lies in borderline or high-risk categories. Similarly, HbA1c levels indicate that the majority of individuals are within the normal range, while smaller segments fall into the prediabetes and diabetes categories. 

<p align="center">
  <img src="Assets/EDA_bp.png" width="71.7%" />
  <img src="Assets/EDA_bmi.png" width="27.3%" />
</p> 

The blood pressure scatter plot illustrates the relationship between systolic and diastolic pressure, highlighting clusters corresponding to normal, elevated, and hypertension stages. The categorized blood pressure distribution shows that normal blood pressure is the most common, followed by hypertension stage 1 and elevated blood pressure, while fewer individuals fall into low blood pressure or hypertension stage 2 categories. Body mass index (BMI) distribution indicates that overweight and normal BMI categories dominate the dataset, with smaller proportions classified as obese, underweight, or severe obesity. 

<p align="center">
  <img src="Assets/EDA_health_condition.png" width="51%" />
  <img src="Assets/EDA_medication_count.png" width="48%" />
</p> 

Among reported health conditions, hypertension is the most prevalent, followed by mental health conditions, arthritis, and diabetes, while conditions such as cancer history, liver disease, and kidney disease occur less frequently. Medication usage patterns show that most individuals take zero to one medication, with the number of individuals decreasing steadily as the medication count increases, indicating that only a small fraction of the population relies on multiple medications. 

![Dashboard](Assets/EDA_visits.png) 

The distribution of doctor visits in the past year shows that most individuals have 0–3 visits, with the frequency declining steadily as the number of visits increases, indicating that frequent medical consultations are relatively uncommon. The hospitalizations over the last three years reveal that the vast majority of individuals have no hospital admissions, while only a small fraction experienced one or more hospitalizations. A similar pattern appears in the number of days hospitalized, where most individuals recorded zero days, and only a very small proportion stayed in the hospital for extended periods, suggesting generally low hospitalization intensity within the population. 

<p align="center">
  <img src="Assets/EDA_annual_medical_cost.png" width="42.5%" />
  <img src="Assets/EDA_plan.png" width="56.5%" />
</p> 

The annual medical cost distribution shows that most individuals fall into the very low or low cost categories, with only a small number reaching medium or high cost levels, indicating that high healthcare expenditures are relatively rare. In terms of insurance plan types, the population is fairly balanced between Preferred Provider Organization (PPO) and Health Maintenance Organization (HMO) plans, while smaller shares belong to Exclusive Provider Organization (EPO) or Point-of-Service (POS) plans. The network tier distribution indicates that Silver plans are the most common, followed by Bronze, while Gold and Platinum plans represent smaller segments of the insured population. 

![Dashboard](Assets/EDA_premium_deductible_copay.png) 

The final visualization set examines insurance pricing and cost-sharing structures. The annual premium distribution shows that most individuals pay lower premium amounts, particularly below $1000 annually, while higher premium categories are progressively less common. The deductible distribution suggests that moderate and low deductible plans dominate, with fewer individuals enrolled in high or very high deductible plans. Lastly, the copay distribution indicates that smaller copay amounts are most common, while higher copay levels appear less frequently, reflecting typical insurance plan designs that aim to balance affordability and cost sharing for policyholders. 

## Model Training 

The cleaned and feature-engineered dataset was used for model training, with `annual_premium` defined as the target variable. The data was split into training and testing sets using an 80–20 ratio with a fixed **random_state=42** to ensure reproducibility. Categorical variables were transformed using One-Hot Encoding, while numerical features were used directly for most models. Since ElasticNet is sensitive to feature scale, numerical features were standardized using StandardScaler within a pipeline. 

Four regression models were trained and evaluated: **XGBoost**, **Random Forest**, **ElasticNet**, and **LightGBM**. Tree-based models (XGBoost, Random Forest, LightGBM) used the encoded features directly, while ElasticNet used a pipeline combining scaling and regression. Model performance was assessed using **MAE**, **MSE**, **RMSE**, and **R²** Score to compare prediction accuracy across models. To avoid compatibility issues during application deployment, the XGBoost model was retrained separately on **CPU in a Google Colab environment**, ensuring stable inference in environments where GPU dependencies may not be available. 

| Model Name    | Implementation                          | Key Parameters Used                                                                               | Training Time |
| ------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------- |
| XGBoost       | `xgb.XGBRegressor`                      | `objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42` | ~5 Seconds    |
| ElasticNet    | `Pipeline(StandardScaler → ElasticNet)` | `random_state=42`                                                                                 | ~5 Seconds    |
| Random Forest | `RandomForestRegressor`                 | `n_estimators=100, random_state=42, n_jobs=-1`                                                    | ~2 Minutes    |
| LightGBM      | `lgb.LGBMRegressor`                     | `n_estimators=100, learning_rate=0.1, random_state=42`                                            | ~6 Seconds    |

## Results 

After training, the models were evaluated on the test dataset using four common regression metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score. Lower values of MAE, MSE, and RMSE indicate better prediction accuracy, while a higher R² score indicates stronger explanatory power of the model. 

![Dashboard](Assets/performance_matrix.png) 

The bar charts above compare the performance of all four models across these metrics. Random Forest clearly achieves the best results with the lowest MAE, MSE, and RMSE, along with the highest R² score (≈0.99), indicating very accurate predictions. LightGBM and XGBoost also perform well, producing relatively low error values and high R² scores (around 0.96). In contrast, ElasticNet shows significantly higher error values and a lower R² score, suggesting that the linear nature of ElasticNet limits its ability to capture the more complex relationships present in the dataset. 

<p align="center">
  <img src="Assets/actual_predicted_comparison.png" width="61.7%" />
  <img src="Assets/model_performance_visualization.png" width="37.3%" />
</p> 

The grouped bar chart compares the actual annual premiums with the predicted premiums from the four models (XGBoost, Random Forest, ElasticNet, and LightGBM) for 10 sample observations. Each sample contains five bars: the true premium value and the corresponding predictions from each model. This visualization helps illustrate how closely each model’s predictions align with the real values at an individual observation level rather than through aggregate metrics. From the chart, Random Forest and LightGBM predictions consistently remain very close to the actual premium values, indicating strong predictive accuracy across most samples. XGBoost also performs well, with predictions generally close to the actual values, though minor deviations can be observed in some cases. In contrast, ElasticNet predictions show larger deviations, particularly for samples with higher premiums, where it tends to either overestimate or underestimate the values. The scatter plot further visualizes prediction quality by comparing actual premiums with predicted premiums. The red diagonal line represents the ideal prediction line, where predicted values perfectly match actual values. Predictions that fall closer to this line indicate higher accuracy. The scatter points show that Random Forest and LightGBM predictions cluster tightly around the ideal line, demonstrating strong predictive performance. XGBoost points are also reasonably close to the line but with slightly greater variation. ElasticNet predictions deviate more from the ideal line, confirming the weaker performance observed in the bar chart metrics. 

## Explainability 

The model’s predictions are interpreted using SHAP (SHapley Additive exPlanations), which explains how each feature contributes to the final prediction. 

![Dashboard](Assets/SHAP_explanation.png) 

The force plot above shows how the model arrived at a predicted annual premium of 1,022. The prediction starts from the base value (average premium) of 581.9, and each feature either increases or decreases the final output. In this example, the prediction is primarily driven upward by the individual's high annual medical cost (6,789), which the model identifies as a strong risk factor. The insurance network tier also contributes to the increase, as plans above the Bronze tier typically have higher premiums. This effect is partially offset by the individual being enrolled in a Silver plan, which has a lower premium compared to higher-tier options. 

## Deployment 

The application is deployed as an interactive web app using Streamlit.

🔗 Live App: https://health-insurance-premium-prediction-system.streamlit.app/ 

The web interface allows users to input policyholder details and receive a predicted annual health insurance premium based on the trained XGBoost regression model. To improve usability and organization, the input fields are divided into two sections: `Personal & Health Details` and `Medical History & Policy Details`. 

<p align="center">
  <img src="Assets/deployment_top.png" width="48.3%" />
  <img src="Assets/deployment_down.png" width="50%" />
</p> 

All categorical and numerical features used during model training were included as input fields in the web application to ensure consistency between training and inference. Several auto-generated and color-coded informational fields were added to improve user understanding of their health profile. These fields are derived from user inputs but are not used by the model for prediction. These fields are `Age Category`, `BMI Category`, `Blood Pressure Category`, `LDL Cholesterol Category`, `HbA1c Category`, `Annual Medical Cost Category` and  `Deductible Level`. These indicators provide users with contextual feedback about their health and policy parameters without influencing the model output. 

During data preprocessing, abbreviated plan types were replaced with their full forms (e.g., PPO → Preferred Provider Organization). However, these full terms may still be unfamiliar to many users. To improve understanding, an auto-generated `Plan Type Explanation` field was included in the interface to briefly describe the selected insurance plan type.

## Practical Applications 

The developed Health Insurance Premium Prediction System has several practical applications in the insurance and healthcare analytics domain. 

- By estimating an individual's expected insurance premium based on demographic, lifestyle, and medical attributes, the system can assist insurance providers in making data-driven pricing decisions and improving transparency in premium calculation. It can also serve as a decision-support tool during the underwriting process, helping insurers quickly evaluate common risk-related factors such as age, BMI, smoking status, chronic diseases, and healthcare utilization patterns. 

- From a customer perspective, the system can function as an interactive premium estimation tool, allowing users to input their personal and health-related information to obtain an estimated annual premium. This improves transparency and helps individuals better understand how different factors (such as smoking habits, BMI category, or existing medical conditions) may influence their insurance costs. Such tools are commonly used in online insurance platforms (such as AdInsure or Bitrix24) to provide quick quotes before policy purchase. 

- Beyond direct pricing estimation, the model can also support risk analysis and policy planning. Insurance companies can use similar predictive systems to analyze population-level risk patterns, design targeted insurance plans, and optimize policy structures such as deductible levels, copay options, and network tiers. Additionally, healthcare analytics teams can use insights from the model to identify key drivers of healthcare costs, which may support preventive healthcare initiatives and cost management strategies. 

## Limitations 

Despite achieving reasonable predictive performance, the project has several limitations related to data characteristics, model constraints, deployment environment, and explainability. These factors should be considered when interpreting the results of the premium estimation system. 

### Feature Bias 

The target variable `annual_premium` shows a very strong correlation with `annual_medical_cost`, which tends to dominate the model’s decision-making process. As a result, other relevant features such as network_tier, income, blood pressure, blood sugar (HbA1c), employment status, and age have comparatively lower influence on the prediction. A feature importance bar chart is included below to illustrate this imbalance. 

<p align="left">
  <img src="Assets/SHAP_feature_importance.png" width="55%" />
</p> 

This feature dominance may introduce bias and reduce the model’s ability to generalize well for unseen policyholders, particularly for individuals without prior medical cost history. 

### Model Size Constraints 

Ensemble models such as Random Forest produced the most competitive predictive performance during testing. However, the trained model resulted in a large pickle file (~674 MB), making it impractical for lightweight deployment environments such as Streamlit Cloud due to memory and storage limitations. 

### Platform Trade-off 

A **Flask-based** web application using HTML and CSS was initially explored to enable greater UI/UX customization compared to Streamlit. However, deployment on **Render** experienced high cold-start latency and slow initial response times, which negatively impacted user experience. As a result, this deployment approach was postponed. To ensure faster interaction and smoother deployment, the application was ultimately deployed using **Streamlit**. While this allowed for easier integration of the machine learning model and faster loading times, it also introduced limitations in frontend customization compared to a full Flask-based interface. Additionally, the application relies on free-tier cloud infrastructure, which imposes constraints on memory usage, CPU resources and concurrent user handling. 

### SHAP Visualization Limitation 

Explainability using SHAP (SHapley Additive exPlanations) was explored to provide insight into how individual features influence predictions. However, Streamlit cannot directly render the interactive SHAP force plot due to its JavaScript dependencies. As a result, fully interactive explainability visualizations were not implemented in the deployed web application. 

## Tools Used 

## Licence 

## Contact 


