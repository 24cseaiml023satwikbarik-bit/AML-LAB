from flask import Flask, render_template, request
import numpy as np
import matplotlib
# Use 'Agg' backend so matplotlib generates images headlessly without window issues
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    equation = None
    mae = None
    mse = None
    rmse = None
    r2 = None
    plot_url = None
    error = None
    x_val = ""
    y_val = ""
    show_actions = False

    if request.method == 'POST':
        # Retrieve input values
        x_val = request.form.get('x_input', '')
        y_val = request.form.get('y_input', '')
        action = request.form.get('action') 

        if action == 'reset':
            # Clear everything and render clean state pointing to Page.html
            return render_template('Page.html', x_val="", y_val="")

        try:
            # Convert space-separated string values into numpy float arrays
            x = np.array([float(i) for i in x_val.split()])
            y = np.array([float(i) for i in y_val.split()])
            n = len(x)

            if n < 2 or len(x) != len(y):
                error = "Error: Ensure both X and Y inputs have matching lengths and at least 2 data points."
            else:
                # Calculate OLS regression coefficients using the algebraic formula:
                # m = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
                sum_x = np.sum(x)
                sum_y = np.sum(y)
                sum_xy = np.sum(x * y)
                sum_x_squared = np.sum(x ** 2)

                denominator = (n * sum_x_squared) - (sum_x ** 2)
                if denominator == 0:
                    error = "Error: Infinite slope! All X values are identical (denominator is 0)."
                else:
                    m = ((n * sum_xy) - (sum_x * sum_y)) / denominator
                    c = (sum_y - (m * sum_x)) / n
                    equation = f"y = {m:.4f}x + {c:.4f}"
                    show_actions = True

                    # Calculate Evaluation Metrics (MAE, MSE, RMSE, R²)
                    y_pred = m * x + c
                    mae = np.sum(np.abs(y - y_pred)) / n
                    mse = np.sum((y - y_pred) ** 2) / n
                    rmse = np.sqrt(mse)

                    y_mean = np.mean(y)
                    ss_total = np.sum((y - y_mean) ** 2)
                    ss_residual = np.sum((y - y_pred) ** 2)
                    
                    if ss_total == 0:
                        r2 = 1.0 if ss_residual == 0 else 0.0
                    else:
                        r2 = 1.0 - (ss_residual / ss_total)

                    # Handle Visualizations
                    if action == 'visualize':
                        plt.figure(figsize=(8, 5))
                        plt.scatter(x, y, color='#3498db', s=80, zorder=5, label='Actual Data Points')
                        plt.plot(x, y_pred, color='#e74c3c', linewidth=2.5, zorder=4, label=f'Regression Line: {equation}')
                        
                        # Draw vertical residual lines connecting data points to the line
                        for xi, yi, y_pred_i in zip(x, y, y_pred):
                            plt.vlines(xi, yi, y_pred_i, colors='#95a5a6', linestyles='dashed', alpha=0.7)
                            
                        plt.title("Linear Regression & OLS Line of Best Fit", fontsize=14, pad=15)
                        plt.xlabel("X Feature Values", fontsize=11)
                        plt.ylabel("Y Target Values", fontsize=11)
                        plt.legend(loc='best')
                        plt.grid(True, linestyle=':', alpha=0.6)

                        # Encode plot to base64 image string so we don't need local image files
                        img = io.BytesIO()
                        plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
                        img.seek(0)
                        plot_url = base64.b64encode(img.getvalue()).decode()
                        plt.close()

        except ValueError:
            error = "Error: Please enter only valid numeric values separated by spaces."

    # Render templates/Page.html instead of index.html
    return render_template(
        'Page.html', 
        equation=equation, 
        mae=mae, 
        mse=mse, 
        rmse=rmse, 
        r2=r2, 
        plot_url=plot_url, 
        error=error, 
        x_val=x_val, 
        y_val=y_val,
        show_actions=show_actions
    )

if __name__ == '__main__':
    app.run(debug=True)