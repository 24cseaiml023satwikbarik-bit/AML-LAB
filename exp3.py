import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_excel('cgpaPRED.xls')

features_multi = ['SEM 1', 'SEM 2', 'SEM 3', 'SEM 4']
feature_simple = ['SEM 4'] 
label = 'SEM 5'


plt.figure(figsize=(8, 5))
sns.boxplot(data=df[features_multi + [label]], palette="Set2")
plt.title("Box Plot of Semester Grades")
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(14, 5))


sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=axes[0])
axes[0].set_title('Correlation Heatmap')

sns.scatterplot(x='SEM 4', y='SEM 5', data=df, ax=axes[1])
axes[1].set_title('SEM 4 vs SEM 5 Scatter')

plt.tight_layout()
plt.show()


y = df[label]

X_train_m, X_test_m, y_train, y_test = train_test_split(df[features_multi], y, test_size=0.2, random_state=42)
multi_model = LinearRegression().fit(X_train_m, y_train)
y_pred_multi = multi_model.predict(X_test_m)

X_train_s, X_test_s, _, _ = train_test_split(df[feature_simple], y, test_size=0.2, random_state=42)
simple_model = LinearRegression().fit(X_train_s, y_train)
y_pred_simple = simple_model.predict(X_test_s)


metrics_df = pd.DataFrame({
    'Model': ['Simple Linear', 'Multiple Linear'],
    'R2 Score': [r2_score(y_test, y_pred_simple), r2_score(y_test, y_pred_multi)],
    'MSE': [mean_squared_error(y_test, y_pred_simple), mean_squared_error(y_test, y_pred_multi)]
})


fig, axes = plt.subplots(1, 3, figsize=(18, 5))


axes[0].scatter(X_test_s, y_test, color='blue', label='Actual', alpha=0.6)
axes[0].plot(X_test_s, y_pred_simple, color='red', linewidth=2, label='Regression Line')
axes[0].set_title("Simple Linear Regression")
axes[0].set_xlabel("SEM 4")
axes[0].set_ylabel("SEM 5")
axes[0].legend()

# Multiple Linear Visualization (Actual vs Predicted line)
axes[1].scatter(y_test, y_pred_multi, color='green', alpha=0.6)
# Plotting a perfect 45-degree fit line for reference
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
axes[1].set_title("Multi-Linear (Actual vs Predicted)")
axes[1].set_xlabel("Actual SEM 5")
axes[1].set_ylabel("Predicted SEM 5")

# Performance Metrics Bar Chart
metrics_df.plot(x='Model', y=['R2 Score', 'MSE'], kind='bar', ax=axes[2], colormap='viridis')
axes[2].set_title("Performance Metrics Comparison")
axes[2].set_xticklabels(metrics_df['Model'], rotation=0)
axes[2].set_ylabel("Score")

plt.tight_layout()
plt.show()