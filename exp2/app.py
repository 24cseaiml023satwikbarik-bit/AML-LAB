from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg') # Required to generate plots without a GUI in Flask
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# 1. Data & Model Training
df = pd.DataFrame({
    'Study_Hours': [2, 4, 6, 8],
    'Attendance': [70, 80, 85, 90],
    'Assignment_Marks': [12, 15, 18, 20],
    'Final_Marks': [45, 60, 75, 90]
})

# Define Features (X) and Target (y)
X = df[['Study_Hours', 'Attendance', 'Assignment_Marks']]
y = df['Final_Marks']

# Train the Model
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    equation = None
    plot_url = None

    if request.method == 'POST':
        try:
            # Fetch inputs from the HTML form
            attendance = float(request.form['attendance'])
            hours = float(request.form['study_hours'])
            assignments = float(request.form['assignment_marks'])

            # 2. Live Prediction (Column names must match the training data X exactly)
            new_student = pd.DataFrame({
                'Study_Hours': [hours], 
                'Attendance': [attendance], 
                'Assignment_Marks': [assignments]
            })
            
            raw_pred = model.predict(new_student)[0]
            # Clamp the prediction between 0 and 100
            prediction = round(min(100.0, max(0.0, raw_pred)), 2)

            # Format the multilinear regression equation line
            coef = model.coef_
            intercept = model.intercept_
            equation = f"Final = {intercept:.2f} + ({coef[0]:.2f} × Study Hours) + ({coef[1]:.2f} × Attendance) + ({coef[2]:.2f} × Assignments)"

            # 3. & 4. Calculate Metrics and Visualize (If button was clicked)
            if 'visualize' in request.form:
                mse = mean_squared_error(y, y_pred)
                metrics = ['MAE', 'RMSE', 'MSE', 'R²']
                values = [mean_absolute_error(y, y_pred), np.sqrt(mse), mse, r2_score(y, y_pred)]

                # Generate side-by-side scatter plots
                fig = plt.figure(figsize=(10, 4))

                # Plot 1: Fit (Actual vs Predicted)
                plt.subplot(121)
                plt.scatter(y, y_pred, color='#3498db')
                plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', label='Perfect Fit')
                plt.title('Actual vs Predicted')
                plt.xlabel('Actual Marks')
                plt.ylabel('Predicted Marks')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.5)

                # Plot 2: Metrics Scatter
                plt.subplot(122)
                plt.scatter(metrics, values, color='orange', s=100)
                for i, val in enumerate(values):
                    plt.annotate(f'{val:.2f}', (metrics[i], values[i]), textcoords="offset points", xytext=(0,5), ha='center')
                plt.title('Evaluation Metrics')
                plt.xlabel('Metrics')
                plt.ylabel('Values')
                plt.grid(True, linestyle='--', alpha=0.5)

                plt.tight_layout()

                # Convert plot to PNG image in base64 to display in HTML
                img = io.BytesIO()
                plt.savefig(img, format='png', bbox_inches='tight')
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode()
                plt.close(fig)

        except ValueError:
            prediction = "Invalid input. Please enter numbers only."

    return render_template('index.html', prediction=prediction, equation=equation, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)