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

## Project Workflow 

![Dashboard](Assets/workflow_corrected.png) 

## Dataset 

[Dataset](https://www.kaggle.com/datasets/mohankrishnathalla/medical-insurance-cost-prediction) 

## Data Cleaning 

## Feature Engineering 

## Exploratory Data Analysis 

The original dataset contains 50+ columns, including several fields such as `person_id`, `claims_count`, `avg_claim_amount`, `proc-imaging_count` and `total_claims_paid`. These variables were excluded because they are either irrelevant for predictive modeling, act as identifiers, or represent post-outcome information that would not be available during real-world predictions. In addition, the variable `is_high_risk` was removed due to data leakage, as it directly reflects information closely related to the prediction target. To ensure meaningful analysis, the exploratory data analysis was conducted using cleaned and feature-engineered dataset (derived from the raw dataset). This allows the analysis to focus only on relevant demographic, socioeconomic, and risk-related features that are actually used by the machine learning models. 

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

## Results 

![Dashboard](Assets/performance_matrix.png) 

<p align="center">
  <img src="Assets/actual_predicted_comparison.png" width="61.7%" />
  <img src="Assets/model_performance_visualization.png" width="37.3%" />
</p> 

## Practical Applications 

## Explainability 

![Dashboard](Assets/SHAP_explanation.png) 

The SHAP force plot shows how the model arrived at a predicted annual premium of 1,022. Starting from the base value (average premium) of 581.9, the prediction was primarily increased by the individuals high annual medical cost (6,789), which the model identifies as a strong risk factor. The selected insurance plan tier also contributed to the increase, as non-Bronze plans tend to have higher premiums. This increase was partially offset by being enrolled in a Silver plan rather than a higher-tier option. Overall, the plot provides a transparent breakdown of how each feature contributed to the final prediction. 

## Deployment 

https://health-insurance-premium-prediction-system.streamlit.app/ 

<p align="center">
  <img src="Assets/deployment_top.png" width="48.3%" />
  <img src="Assets/deployment_down.png" width="50%" />
</p> 

## Limitations 

![Dashboard](Assets/SHAP_feature_importance.png) 

## Tools Used 

## Licence 

## Contact 


