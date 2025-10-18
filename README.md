# Edge-AI-Development

## When You Hear "AI" Today...

Artificial Intelligence today often means **large, cloud-based systems** powered by massive GPU clusters and internet-connected models.  
They generate essays, explain physics, create deepfakes, and run on remote servers rather than your device.

Typical examples include:
- ChatGPT writing essays  
- Gemini explaining physics  
- Deepfakes selling crypto  
- Cloud GPUs, massive models, internet-dependent execution  

These systems are incredibly capable—but they come with **high energy costs, latency, and privacy challenges**.  
Most “AI” today happens **in the cloud**, far from the edge where real-time, low-power intelligence is needed.

This repository explores the next step: bringing AI **closer to the hardware**—efficient, local, and adaptive.

## Programming: Logic vs Learning

Programming paradigms are evolving.  
Traditional programming relies on **explicit logic** — you write every rule, and the code behaves exactly as instructed.  
Artificial Intelligence, on the other hand, depends on **learning from data** — the model infers the rules by observing examples.

| Traditional Logic | AI / Machine Learning |
|--------------------|------------------------|
| You write the rules | You feed it examples |
| Code is predictable | It learns patterns |
| Like RISC-V assembly: clean, minimal, direct | Like a toddler learning by trial and error |

While traditional logic offers precision and determinism, learning-based systems introduce adaptability and pattern recognition — ideal for uncertain or dynamic environments.

This repository bridges both worlds — combining **logical structure** with **learning-driven adaptability** for next-generation intelligent systems.

## AI on RISC-V: Brains Without Comfort

Running AI on **RISC-V** is like giving intelligence to a system with no safety net —  
no Linux, no Python, and minimal memory. Yet that’s exactly what makes it exciting.

<img width="578" height="328" alt="image" src="https://github.com/user-attachments/assets/047f7f2f-5680-4e7d-b8c3-db1da080683d" />

### Challenges
- No Linux, no Python, no high-level frameworks  
- Low RAM and low power, but full hardware control  
- Feels like teaching a rock to play chess — slow at first, but it works

| Feature | Raspberry Pi | RISC-V (VSD Pro) |
|----------|---------------|------------------|
| **OS** | Full Linux | Bare-metal |
| **Tools** | Python, PyTorch | C, Assembly |
| **AI-Ready?** | ✅ Yes | ❌ Not yet (but we’ll change that) |

The mission of this project is to **bring AI inference to bare-metal RISC-V boards**,  
proving that intelligent behavior doesn’t need a data center — just efficient design.

# Understanding RISC-V Board

## What This Project Is Really About

This project is designed to **build AI from the ground up** — not in the cloud, but directly on RISC-V boards and bare-metal systems.

### Objectives
- **Understand AI from first principles:** learn how intelligence emerges from logic and computation.  
- **Deploy models on real hardware:** bring neural networks to RISC-V microcontrollers and embedded boards.  
- **Embrace true constraints:** memory limits, timing accuracy, and power optimization.  
- **Build efficient edge intelligence:** create tiny AI systems that *think* locally — no GPU, no Python runtime, no internet dependency.

### Why It Matters
While most modern AI runs in massive data centers, this project explores the *opposite end of the spectrum* —  
**minimalist AI that runs where data originates**. It’s about understanding, optimizing, and reshaping what “intelligence” means at the hardware level.

---

# Understanding Processor Clock Speed and Inference Performance

## 1. What Is Clock Speed?

A **processor clock speed** (e.g., **320 MHz**) represents how many cycles (ticks) the CPU performs per second.

- **1 Hz = 1 cycle per second**  
- **320 MHz = 320 million cycles per second**

Each cycle corresponds to a step in which the CPU can execute part (or all) of an instruction.  
Hence, the clock speed defines the **upper limit** of how fast computations can occur.

---

## 2. Instructions and Clock Cycles

Every instruction—addition, multiplication, memory load, etc.—requires a certain number of **clock cycles** to execute.

| Instruction Type | Example | Typical Cycles |
|------------------|----------|----------------|
| Integer Addition | `ADD R1, R2` | 1 |
| Multiplication   | `MUL R3, R4` | 3–4 |
| Memory Access    | `LOAD R5, [R1]` | 5–10 |
| Branch           | `IF … GOTO` | 2–3 |

Since **1 cycle = 1 / 320 × 10⁶ s ≈ 3.125 ns**,  
an operation taking 4 cycles ≈ 12.5 ns,  
allowing roughly **80 million multiplications per second** (≈ 320 MHz / 4).

---

## 3. Relating Clock Speed to Inference Time

An **inference** (e.g., forward pass of a neural network) consists of many such basic operations (commonly MAC = Multiply–Accumulate).

Assume:

- Model requires **100 million MACs**  
- Each MAC takes **4 cycles**
- Processor speed = **320 MHz**

\[
\text{Time per inference} = \frac{100\,\text{M ops} \times 4\,\text{cycles/op}}{320\,\text{M cycles/s}} = 1.25\,\text{s}
\]

Therefore, one inference ≈ **1.25 seconds**.

---

## 4. Modern Enhancements

Modern CPUs employ several features to improve performance beyond raw frequency:

- **Pipelining:** Overlaps instruction stages → lowers effective cycles per instruction (CPI < 1)  
- **SIMD / Vector Units:** Execute multiple operations per cycle (e.g., 4 or 8 MACs at once)  
- **Caching:** Reduces delays for memory fetches  

If the processor executes **8 MACs per cycle**, the throughput becomes:

\[
320\,\text{MHz} \times 8 = 2.56\,\text{G MAC/s}
\]

leading to much faster inference.

---

## 5. Simplified Performance Formula

| Parameter | Meaning | Effect |
|------------|----------|--------|
| **Clock Speed** | Cycles per second | ↑ → Faster |
| **Cycles per Instruction (CPI)** | Ticks per instruction | ↓ → Faster |
| **Instructions per Cycle (IPC)** | Instructions executed per tick | ↑ → Faster |
| **Workload** | Operations per inference | ↑ → Slower |

Overall performance can be approximated as:

\[
\text{Inference Time} = \frac{\text{Operations per Inference} \times \text{CPI}}{\text{Clock Speed} \times \text{IPC}}
\]

---

## 6. Example Comparison

| Processor Speed | Cycles per Op | Total Ops | Total Time ≈ |
|-----------------|---------------|------------|---------------|
| 100 MHz | 4 | 100 M | 4.00 s |
| 320 MHz | 4 | 100 M | 1.25 s |
| 1 GHz | 4 | 100 M | 0.40 s |

Higher frequency or better instruction efficiency = faster inference.

---

### Key Insight

Clock speed defines **how many opportunities per second** your processor has to perform work.  
The **real speed** of inference depends on how many cycles each instruction takes and how efficiently the CPU uses each tick.

---

# Best-Fitting Lines 101 - Getting Started With ML

## Linear Regression Visualization

<img width="1917" height="847" alt="image" src="https://github.com/user-attachments/assets/adeee273-d21f-4d0a-866c-3318b541753e" />

This visualization demonstrates a **simple linear regression** model built from **11 data points**.  
The regression line approximates the relationship between input variable \( x_1 \) and output \( y_1 \).

### Key Results
- **Regression Line:** \( y = 0.4894x + 5.4764 \)  
- **Number of Data Points:** 11  
- **Sum of Squared Errors (SSE):** 16.9785  
- **Total Sum of Squares (SST):** 48.3627  
- **Coefficient of Determination (\( R^2 \)) = 0.8056**  
- **Correlation Coefficient (r) = 0.8975**  
- **Mean Squared Error (MSE) = 1.5435**  
- **Prediction Error = 0.1308**

### Interpretation
The model achieves a strong correlation (\( r = 0.89 \)) and an \( R^2 \) value of approximately 0.81,  
indicating that around **81% of the variance in \( y_1 \)** can be explained by \( x_1 \).  
The **prediction error** of 0.13 reflects a small deviation between the actual and predicted data points.

Each red point (\( x_i \)) represents an individual observation, while the blue line shows the model’s prediction.  
Vertical dashed lines indicate **residuals**, or how far each data point deviates from the regression fit.

### Purpose in This Project
This regression model serves as an **introductory experiment** in the broader goal of understanding AI  
from first principles — starting with classic statistical learning before scaling down models  
for **embedded and RISC-V edge hardware**. It demonstrates:
- How learning replaces rule-based logic.  
- The role of error and feedback in improving models.  
- The foundation upon which neural networks and adaptive systems are built.

## 🔗 Interactive Regression Model

You can explore the live interactive version of the regression model here:  
[View on Desmos](https://www.desmos.com/calculator/ylifeiyfqw)

---

# Gradient Descent Unlocked

# Linear Regression from Scratch using Python

This project demonstrates how to build a **Linear Regression model from scratch** using **NumPy**, **Pandas**, and **Matplotlib** — without using prebuilt machine learning libraries like scikit-learn.

The model predicts students’ exam scores based on the number of hours they studied and explains how **Gradient Descent** works to minimize error and find the best-fit line.

---

## Objective

We aim to find a straight-line relationship between the number of hours studied and the corresponding exam scores.

\[
Y = mX + b
\]

Where:  
- **X** → Hours studied  
- **Y** → Scores achieved  
- **m** → Slope (weight)  
- **b** → Intercept (bias)

Our goal is to determine the best values of **m** and **b** that minimize the difference between predicted and actual results.

---

## Dataset

Create a file named **`student_scores.csv`** with the following content:

```csv
Hours,Scores
1.1,17
2.5,21
3.2,27
4.5,30
5.5,40
6.8,45
8.2,50
9.5,60
10.5,75
12,85
```

## Full Code
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read dataset
dataset = pd.read_csv('student_scores.csv')

# Scatter plot
plt.scatter(dataset['Hours'], dataset['Scores'])
plt.title("Study Hours vs Scores")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.show()

# Extracting features and labels
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values
print(X)

# Define Model class
class Model():
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def predict(self, X):
        return X.dot(self.slope) + self.const

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.slope = np.zeros(self.n)
        self.const = 0
        self.X = X
        self.Y = Y

        for i in range(self.iterations):
            self.update_weights()
        return self

    def update_weights(self):
        Y_pred = self.predict(self.X)
        dw = - (2 * (self.X.T).dot(self.Y - Y_pred)) / self.m
        db = - 2 * np.sum(self.Y - Y_pred) / self.m
        self.slope = self.slope - self.learning_rate * dw
        self.const = self.const - self.learning_rate * db

# Create and train model
model = Model(learning_rate=0.01, iterations=1000)
model.fit(X, Y)

# Predict values
Y_pred = model.predict(X)
print("Predicted Values:")
print(Y_pred)

# Plot results
plt.scatter(dataset['Hours'], dataset['Scores'], label="Actual Data")
plt.plot(X, Y_pred, color='red', label="Regression Line")
plt.title("Linear Regression Fit")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.legend()
plt.show()

```

### 1. Hypothesis Function

The hypothesis (or prediction) equation of a linear regression model is:

$$
\hat{Y} = mX + b
$$

Where:  
- \( \hat{Y} \) → Predicted output (score)  
- \( X \) → Input feature (hours studied)  
- \( m \) → Weight or slope of the line  
- \( b \) → Bias or intercept of the line  

The model tries to find values of \( m \) and \( b \) that minimize prediction errors.

---

### 2. Cost Function — Mean Squared Error (MSE)

The cost function \( J(m, b) \) measures how well the line fits the data.  
It computes the average squared difference between actual and predicted values.

$$
J(m, b) = \frac{1}{N} \sum_{i=1}^{N} \left( Y_i - (mX_i + b) \right)^2
$$

Where:  
- \( N \) → Number of data points  
- \( Y_i \) → Actual value for sample *i*  
- \( mX_i + b \) → Predicted value for sample *i*  

The smaller \( J(m, b) \) is, the better the model fits the data.

---

### 3. Gradient Descent Optimization

Gradient Descent is used to minimize the cost function.  
It works by updating parameters \( m \) and \( b \) in the direction of the negative gradient.

#### Partial Derivatives of the Cost Function:

For slope \( m \):

$$
\frac{\partial J}{\partial m} = -\frac{2}{N} \sum_{i=1}^{N} X_i \left( Y_i - \hat{Y_i} \right)
$$

For intercept \( b \):

$$
\frac{\partial J}{\partial b} = -\frac{2}{N} \sum_{i=1}^{N} \left( Y_i - \hat{Y_i} \right)
$$

#### Parameter Update Rules:

After computing gradients, update \( m \) and \( b \) as follows:

$$
m = m - \alpha \frac{\partial J}{\partial m}
$$

$$
b = b - \alpha \frac{\partial J}{\partial b}
$$

Where:  
- \alpha → Learning rate (controls the step size of updates)  
- The process repeats for multiple iterations until the cost \( J(m,b) \) converges to a minimum value.  

---

### 4. Mean Squared Error Simplified (Expanded Form)

To understand why the error decreases, expand the cost function:

$$
J(m,b) = \frac{1}{N} \sum_{i=1}^{N} \left( Y_i^2 - 2Y_i(mX_i + b) + (mX_i + b)^2 \right)
$$

Each update step reduces this cost by adjusting \( m \) and \( b \) in small increments controlled by \( \alpha \).

---

### 5. Convergence Condition

Training stops when:

$$
|J_{current} - J_{previous}| < \varepsilon
$$

Where \( \varepsilon \) is a small threshold (e.g., 0.0001), indicating that the error no longer decreases significantly.

---

### 6. Final Regression Equation

Once training completes, the best-fit line is obtained:

$$
\boxed{\hat{Y} = mX + b}
$$

This line can now be used to make predictions for unseen data.

---

### 7. Example: Parameter Update Visualization

Iteration step:

$$
m_{new} = m_{old} - \alpha \frac{\partial J}{\partial m}
$$

$$
b_{new} = b_{old} - \alpha \frac{\partial J}{\partial b}
$$

Over many iterations, the values of \( m \) and \( b \) converge to those that minimize the mean squared error.

---

### 8. Cost Function Graph

The cost function \( J(m, b) \) forms a **convex surface** (a bowl shape).  
Gradient Descent ensures that starting from any initial value, it will eventually reach the **global minimum**.

---

# Predicting Startup Profits - Multiple Linear Regression

This repository demonstrates a **multiple linear regression model** applied on the `50_Startups.csv` dataset using **Python** and **scikit-learn**.  
It builds a regression model that predicts company profit based on R&D Spend, Administration, and Marketing Spend.

---

## Overview

This project covers:
1. Loading and cleaning the dataset  
2. Splitting data into training and test sets  
3. Building a Multiple Linear Regression model  
4. Making predictions and comparing results  
5. Displaying coefficients, intercept, and regression equation  

---

## Dataset Information

- **Dataset file:** `50_Startups.csv`  
- **Expected columns:**
  - `R&D Spend` — research & development investment  
  - `Administration` — admin cost  
  - `Marketing Spend` — marketing budget  
  - `State` — categorical (removed for this implementation)  
  - `Profit` — target variable (dependent)

> Ensure `50_Startups.csv` is located in the same folder as your Python script.

---

## Requirements

Install dependencies before running the script:

```bash
pip install pandas scikit-learn
```
---

## Code

```
# Multiple Linear Regression Model
# Using 50_Startups.csv Dataset

# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 2: Load the dataset
dataset = pd.read_csv('50_Startups.csv')

# Step 3: Remove the 'State' column (non-numeric)
dataset = dataset.drop('State', axis=1)

# Step 4: Separate the independent (X) and dependent (Y) variables
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Step 5: Split the dataset into Training and Testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Step 6: Create and train the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Step 7: Make predictions on the test set
y_pred = regressor.predict(X_test)
print("\nPredicted Values:")
print(y_pred)

# Step 8: Compare predicted and actual values
print("\nComparison of Actual vs Predicted:")
for i, (pred, actual) in enumerate(zip(y_pred, Y_test)):
    print(f"Sample {i+1}: Predicted = {pred:.2f}, Actual = {actual:.2f}")

# Step 9: Display model coefficients and intercept
print("\nModel Coefficients:", regressor.coef_)
print("Model Intercept:", regressor.intercept_)

# Step 10: Display the final regression equation
print("\nRegression Equation:")
print(f"Y = {regressor.coef_[0]:.7f}*x1 + {regressor.coef_[1]:.7f}*x2 + {regressor.coef_[2]:.7f}*x3 + {regressor.intercept_:.7f}")

```
---

## Sample Output

```
Predicted Values:
[103015.20 132582.28 133222.89  72919.10 179720.10 115049.64  67429.92  98581.96 114822.49 168537.38]

Comparison of Actual vs Predicted:
Sample 1: Predicted = 103015.20, Actual = 103282.38
Sample 2: Predicted = 132582.28, Actual = 144259.40
Sample 3: Predicted = 133222.89, Actual = 146121.95
...

Model Coefficients: [ 0.80571614  0.00109725  0.02639357]
Model Intercept: 49032.89914125249

Regression Equation:
Y = 0.8057161*x1 + 0.0010973*x2 + 0.0263936*x3 + 49032.8991413

```

---

## Explanation

- **X (Independent Variables):** R&D Spend, Administration, Marketing Spend  
- **Y (Dependent Variable):** Profit  

- **Coefficients (`coef_`):** Indicate how much `Y` changes when each `X` changes.  
- **Intercept (`intercept_`):** Value of `Y` when all inputs are zero.  

**Final Model Equation Format:**

\[
Y = a_1x_1 + a_2x_2 + a_3x_3 + b
\]

Where:  
- \( x_1 \) = R&D Spend  
- \( x_2 \) = Administration  
- \( x_3 \) = Marketing Spend  
- \( Y \) = Predicted Profit  
- \( a_1, a_2, a_3 \) = Model Coefficients  
- \( b \) = Intercept  

---

# Degree Up - Fitting Complex Patterns for Edge AI - Polynomial Regression

This project shows how to fit a **polynomial regression** model using scikit-learn by expanding inputs with `PolynomialFeatures` and then training a standard `LinearRegression`.  
We’ll use the classic `50_Startups.csv` dataset, drop the non-numeric `State` column, generate **degree=3** polynomial features, train, and visualize predictions.

---

## 📦 Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- scikit-learn

Install:
```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## Full Code
```
# Polynomial Regression Example (degree=3) on 50_Startups.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 1) Load the dataset
dataset = pd.read_csv('50_Startups.csv')

# 2) Remove the non-numeric 'State' column
dataset = dataset.drop('State', axis=1)

# 3) Separate features (X) and target (Y)
#    X: [R&D Spend, Administration, Marketing Spend]
#    Y: Profit
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# 4) Create polynomial features (degree = 3)
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
print("Transformed feature shape:", X_poly.shape)   # (n_samples, n_poly_terms)

# 5) Fit a linear model on polynomial features
model = LinearRegression()
model.fit(X_poly, Y)

# 6) Inspect learned parameters
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# 7) Predict on the same X (for demo/visualization)
y_pred = model.predict(X_poly)
print("Predictions (first 10):", y_pred[:10])

# 8) Simple visualization
# NOTE: X has 3 columns. For a quick 2D scatter, we plot against the first feature (R&D Spend).
x_plot = X[:, 0] if X.ndim > 1 else X
order = np.argsort(x_plot)           # sort for a cleaner curve
x_plot_sorted = x_plot[order]
y_pred_sorted = y_pred[order]
y_true_sorted = Y[order]

plt.scatter(x_plot_sorted, y_true_sorted, color='blue', label='Actual')
plt.plot(x_plot_sorted, y_pred_sorted, color='red', label='Predicted (poly deg=3)')
plt.title('Polynomial Regression (Profit vs R&D Spend)')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.legend()
plt.tight_layout()
plt.show()

```

---

## 🧮 Mathematical Explanation

### 1. Hypothesis Function

Polynomial Regression models the target variable as:

**Ŷ = β₀ + β₁X + β₂X² + β₃X³ + ... + βₙXⁿ**

For multiple input variables:

**Ŷ = β₀ + β₁x₁ + β₂x₂ + β₃x₃ + β₄x₁² + β₅x₁x₂ + β₆x₃² + ...**

**Where:**
- **Ŷ** → Predicted output  
- **(x₁, x₂, x₃)** → Independent variables  
- **βᵢ** → Model coefficients (weights)

---

### 2. Cost Function

The objective of the model is to minimize the **Mean Squared Error (MSE):**

**J(β) = (1/N) × Σ(Yᵢ − Ŷᵢ)²**

**Where:**
- **Yᵢ** → Actual (true) value  
- **Ŷᵢ** → Predicted value  
- **N** → Number of samples  

This measures the average squared difference between predicted and actual outputs.

---

### 3. Parameter Optimization (Normal Equation)

The best-fit coefficients are found analytically using the **Normal Equation:**

**β = (XᵀX)⁻¹XᵀY**

This computes optimal weights by minimizing the cost function directly.  
In Scikit-Learn, this step is automatically handled by the `LinearRegression()` model.

---

### 4. Polynomial Transformation

Given the original features:

**X = [x₁, x₂, x₃]**

For a **polynomial degree of 3**, the transformed feature vector is:

**Φ(X) = [1, x₁, x₂, x₃, x₁², x₂², x₃², x₁x₂, x₁x₃, x₂x₃, x₁³, x₂³, x₃³, ...]**

Then, the new hypothesis function becomes:

**Ŷ = θ₀ + θ₁Φ₁(X) + θ₂Φ₂(X) + ... + θₖΦₖ(X)**

This allows the linear regression model to learn **nonlinear relationships** using polynomially expanded features.

---

### 5. Example Model Equation

After training, a possible model could look like:

**Y = 0.7788x₁ + 0.0294x₂ + 0.0347x₃ + 42989.0082**

**Where:**
- **x₁** → R&D Spend  
- **x₂** → Administration  
- **x₃** → Marketing Spend  

**Interpretation:**
- For every increase in R&D spend (**x₁**), profit increases by ~0.7788 units.  
- For every increase in Administration spending (**x₂**), profit increases by ~0.0294 units.  
- For every increase in Marketing Spend (**x₃**), profit increases by ~0.0347 units.  
- **42989.0082** is the **intercept (base profit)** when all inputs are zero.

---

### 📊 Summary

- Polynomial Regression generalizes Linear Regression by adding polynomial terms of input features.  
- It fits a **curved surface** to the data, making it suitable for **nonlinear relationships**.  
- Scikit-Learn’s `PolynomialFeatures` automates creation of these polynomial terms.  

**Final Model:**  
**Ŷ = β₀ + β₁x₁ + β₂x₂ + β₃x₃ + ... + βₙxₙ**

<img width="1342" height="750" alt="image" src="https://github.com/user-attachments/assets/f5e766ef-5691-4d3f-b295-0eac462fe565" />


---

# From Python to Silicon - Model on RISC-V

```
/* Copyright 2019 SiFive, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#include <stdio.h>
#include <metal/cpu.h>
#include <metal/led.h>
#include <metal/button.h>
#include <metal/switch.h>

#define RTC_FREQ        32768

#define x1 0.77884104f
#define x2 0.0293919f
#define x3 0.03471025f
#define b 42989.00816508669f

float predict(float inp1, float inp2, float inp3){
    return x1*inp1 + x2*inp2 + x3*inp3 + b;
}

void print_float(float val){
    int int_part = (int)val;
    int frac_part = (int)((val - int_part) * 100);  // 2 decimal places
    if (frac_part < 0) frac_part *= -1;
    printf("%d.%02d", int_part, frac_part);
}

int main (void)
{
    float RDSpend=165349.2f;
    float ADSpend=136897.9f;
    float MKSpend=471784.1f;
    float profit;
    profit=predict(RDSpend,ADSpend,MKSpend);
    printf("profit is : %f", profit );
    print_float(profit);
    // return
    return 0;
}

```
## Example Simulation: STM 32 Microcontroller
<img width="1918" height="988" alt="image" src="https://github.com/user-attachments/assets/31ec5539-b985-451d-9669-07497900677a" />

---

# From Regression to Classification

This project demonstrates **Logistic Regression** — a supervised machine learning algorithm used for **binary classification** problems.  
We use the dataset **`Social_Network_Ads.csv`**, which predicts whether a user purchases a product based on their **Age** and **Estimated Salary**.

## Code

```
# Step 1: Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Step 2: Importing Dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [1, 2]].values
y = dataset.iloc[:, 3].values

print(X)
print(y)

# Step 3: Splitting Dataset into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Step 4: Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Step 5: Training Logistic Regression Model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Step 6: Making Predictions
y_pred = classifier.predict(X_test)

# Step 7: Evaluating the Model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

print("Coefficients:", classifier.coef_)
print("Intercept:", classifier.intercept_)

# Step 8: Visualization of Training Results
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train, palette={0: 'blue', 1: 'red'}, marker='o')

plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.title("Logistic Regression (Training set)")
plt.show()

```
---

## Mathematical Representation

1. Hypothesis Function

ℎ𝜃(𝑥) = 1 / (1 + 𝑒^(−(𝜃₀ + 𝜃₁𝑥₁ + 𝜃₂𝑥₂)))

where:

ℎ𝜃(𝑥) → predicted probability of belonging to class 1  
𝑥₁, 𝑥₂ → input features (Age and Salary)  
𝜃₀, 𝜃₁, 𝜃₂ → model coefficients  

---

2. Decision Boundary Equation

The logistic regression decision boundary is given by:

𝑝 = 𝜃₀ + 𝜃₁𝑥₁ + 𝜃₂𝑥₂ = 0

For your trained model, the coefficients are:

𝑝 = 2.07665837𝑥₁ + 1.11088221𝑥₂ − 0.95217247 = 0

Rearranging for 𝑦 (Estimated Salary):

1.11088221𝑦 = −2.07665837𝑥 + 0.95217247  
𝑦 = (−2.07665837𝑥 + 0.95217247) / 1.11088221


---

# Implementing KNN Classifier in Python - Smarter Decision Boundaries

## Mathematical Representation

1. Distance Calculation

For a given test point, the distance to each training point is calculated using Euclidean distance:

d = √((x₁ - x₂)² + (y₁ - y₂)²)

Here,
x₁, y₁ → coordinates of the test point  
x₂, y₂ → coordinates of a training point  

---

2. Finding Nearest Neighbors

After calculating all distances, the K points with the smallest distances are selected.  
In this project, K = 5.

---

3. Majority Voting

The predicted class is determined by the majority vote among the K nearest neighbors:

Predicted Class = Mode(y₁, y₂, …, yₖ)

---

4. Decision Boundary

The decision boundary is the region where two classes have equal probability based on K nearest points.

Σ(1/dᵢ) * I(yᵢ = 1) = Σ(1/dᵢ) * I(yᵢ = 0)

Points on this boundary are shown by the color transition in the contour plot.

---

5. Complexity

Each prediction requires computing distances to all training samples:

Time Complexity: O(n × d)

where  
n = number of training samples  
d = number of features  

---

6. In this code:

K = 5  
Distance metric = Euclidean distance  
Output = Predicted label (0 or 1)  
Visualization = Contour region with red and blue clusters

---

# From KNN to SVM - Smarter Models for Embedded Boards














