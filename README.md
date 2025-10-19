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

## Mathematical Representation

1. Decision Function

The Support Vector Machine (SVM) tries to find the best separating boundary between two classes by maximizing the margin.  
The decision function is given by:

f(x) = w₀ + w₁x₁ + w₂x₂

where  
w₀ → bias term  
w₁, w₂ → model weights  
x₁, x₂ → input features (Age and Salary)

---

2. Classification Rule

The SVM predicts the class of a point based on the sign of the decision function:

If f(x) ≥ 0 → Class 1  
If f(x) < 0 → Class 0

Thus, the decision boundary is defined where f(x) = 0.

---

3. Margin and Support Vectors

The margin is the distance between the decision boundary and the closest data points from each class.  
These closest points are called **Support Vectors**.

The margin distance is given by:

Margin = 2 / ||w||

where ||w|| is the magnitude of the weight vector.

---

4. Optimization Objective

The goal of SVM is to maximize the margin (minimize ||w||) while correctly classifying the samples.  
This is done by solving the following optimization problem:

Minimize: (1/2) ||w||²  
Subject to: yᵢ (w·xᵢ + b) ≥ 1,  for all i

where  
yᵢ → true label of sample i (+1 or -1)  
xᵢ → input feature vector  
b → bias term

---

5. RBF (Radial Basis Function) Kernel

In this code, the kernel used is RBF (non-linear).  
The RBF kernel transforms data into a higher-dimensional space to make it linearly separable.

The kernel function is defined as:

K(xᵢ, xⱼ) = exp(−γ ||xᵢ − xⱼ||²)

where  
γ (gamma) controls how much influence a single training example has.  
Higher γ → smaller decision regions, can lead to overfitting.  
Lower γ → smoother boundary, can underfit.

---

6. Final Decision Function

For non-linear SVM with RBF kernel:

f(x) = Σ αᵢ yᵢ K(xᵢ, x) + b

where  
αᵢ → Lagrange multipliers  
K(xᵢ, x) → kernel function between support vector xᵢ and input x  
b → bias term  

The predicted class is determined by the sign of f(x).

---

7. Visualization

The colored contour region represents the decision boundary created by the SVM.  
The red and green regions show how the classifier separates the two classes (Purchased and Not Purchased) based on Age and Estimated Salary.

---

# Deploying SVM Models on RISC-V Boards - From Python to C (Need Board)

## Exporting Scaler Parameters to C Header File

1. Extract Mean and Scale

The `StandardScaler` in Scikit-learn computes two main parameters for each feature:

mean = sc.mean_  
scale = sc.scale_

Here,
- mean → average value of each feature  
- scale → standard deviation of each feature (used for normalization)

---

2. File Creation

We open a new header file named `scaler.h` to store these parameters:

with open("scaler.h", "w") as f:

---

3. Writing Parameters

We define the number of features and write both the mean and scale values into C-style arrays.

#define NUM_FEATURES len(mean)

double mean[NUM_FEATURES] = { m₁, m₂, m₃, … };  
double scale[NUM_FEATURES] = { s₁, s₂, s₃, … };

---

4. Purpose

These parameters allow the same normalization process to be replicated on hardware or embedded systems, ensuring that the input data is scaled in the same way as during model training.

---

5. Example Output (scaler.h)

#define NUM_FEATURES 2

double mean[NUM_FEATURES] = { 35.1234567890, 75000.9876543210 };
double scale[NUM_FEATURES] = { 10.4567891234, 20000.6543219876 };

---

Exported scaler parameters to `scaler.h`
---

## Bare Metal Code

```
#include <stdio.h>
#include "svm_model.h"
#include "scaler.h"

void scale_input(float *x) {
    for (int i = 0; i < NUM_FEATURES; ++i) {
        x[i] = (x[i] - mean[i]) / scale[i];
    }
}

int predict(float *x) {
    float max_score = -1e9;
    int best_class = -1;

    for (int c = 0; c < NUM_CLASSES; ++c) {
        float score = bias[c];
        for (int i = 0; i < NUM_FEATURES; ++i) {
            score += weights[c][i] * x[i];
        }

        if (score > max_score) {
            max_score = score;
            best_class = c;
        }
    }

    return best_class;
}

int main() {
    float input[2] = {19, 90000}; // Example input features

    // Preprocess (feature scaling)
    scale_input(input);

    // Predict class
    int prediction = predict(input);

    // Output result (UART or onboard console)
    printf("Predicted class: %d\n", prediction);

    return 0;
}


```

## Embedded SVM Classification (C Implementation)

1. **Header Imports**

The program uses:
- `svm_model.h` → contains trained model weights and biases
- `scaler.h` → contains mean and scale values from the StandardScaler
- `<stdio.h>` → standard C I/O library for printing results

---

2. **Feature Scaling Function**

Before prediction, features are normalized to match training distribution:

x[i] = (x[i] - mean[i]) / scale[i]

This ensures consistency with Python preprocessing.

---

3. **Prediction Function**

For each class, a decision score is computed as:

score = bias[c] + Σ(weights[c][i] × x[i])

The class with the maximum score is chosen as the prediction.

best_class = argmax(score)

---

4. **Main Execution**

Steps:
- Load input feature vector (e.g., [Age, Salary])
- Scale inputs with `scale_input()`
- Run `predict()` to compute decision scores
- Print predicted class index

---

5. **Example Output**

Input:
Age = 19  
Estimated Salary = 90000  

Output:
Predicted class: 1

---

This code runs efficiently on embedded systems and microcontrollers such as SiFive RISC-V or ARM-based SoCs.

# Embedded SVM Inference on Arduino (Wokwi Simulation)

This project demonstrates how to deploy a trained **Support Vector Machine (SVM)** model on an embedded device using **Arduino** and the **Wokwi Simulator**.  
The trained SVM model parameters (weights, biases, and scaling) are exported from Python and used for real-time inference on a microcontroller.

---

## Overview

This project shows how to:
- Train an SVM in Python (using scikit-learn)
- Export model weights, biases, and scaler parameters
- Deploy the model as a C program running on Arduino hardware (or simulated in Wokwi)

It enables **lightweight edge AI inference** without any external libraries or frameworks.

---

## 📁 Project Structure

├── sketch.ino → Main program (C inference logic)
├── svm_model.h → Model weights and biases
├── scaler.h → Mean and scale values for input normalization


---

## ⚙️ Step 1: Create `svm_model.h`

This header defines the trained SVM model coefficients and bias terms.

```c
#ifndef SVM_MODEL_H
#define SVM_MODEL_H

#define NUM_CLASSES 2
#define NUM_FEATURES 2

// Example model coefficients (replace with your trained values)
const float weights[NUM_CLASSES][NUM_FEATURES] = {
    { 0.45f, 1.23f },
    { -0.65f, 0.98f }
};

// Bias for each class
const float bias[NUM_CLASSES] = { 0.10f, -0.20f };

#endif
```

## Step 2: Create scaler.h

This file contains the mean and scale parameters used to normalize input features during preprocessing.

```
#ifndef SCALER_H
#define SCALER_H

#define NUM_FEATURES 2

// Replace these values with those from your StandardScaler in Python
const float mean[NUM_FEATURES]  = { 35.0f, 75000.0f };
const float scale[NUM_FEATURES] = { 10.0f, 20000.0f };

#endif

```

## Step 3: Create sketch.ino

This Arduino program performs feature scaling, runs SVM inference, and prints the result to the serial monitor.

```
#include <Arduino.h>
#include "svm_model.h"
#include "scaler.h"

// Scale inputs using mean and scale values
void scale_input(float *x) {
  for (int i = 0; i < NUM_FEATURES; ++i) {
    x[i] = (x[i] - mean[i]) / scale[i];
  }
}

// Predict class based on linear SVM decision function
int predict(const float *x) {
  float max_score = -1e9f;
  int best_class = -1;

  for (int c = 0; c < NUM_CLASSES; ++c) {
    float score = bias[c];
    for (int i = 0; i < NUM_FEATURES; ++i) {
      score += weights[c][i] * x[i];
    }
    if (score > max_score) {
      max_score = score;
      best_class = c;
    }
  }
  return best_class;
}

void setup() {
  Serial.begin(115200);

  // Example input features (Age, Salary)
  float input[NUM_FEATURES] = { 19.0f, 90000.0f };

  // Step 1: Scale input
  scale_input(input);

  // Step 2: Predict class
  int cls = predict(input);

  // Step 3: Display result
  Serial.print("Predicted class: ");
  Serial.println(cls);
}

void loop() {
  // Nothing here
}

```

## Example Output

<img width="1912" height="982" alt="image" src="https://github.com/user-attachments/assets/fb7783d0-e961-49f2-8262-1e8876318222" />

---

## Rather than importing one variable at a time we can improt all the variables at one go

## RISC-V Code
```
#include <stdio.h>
#include <math.h>
#include "svm_model.h"
#include "scaler.h"

// Function to scale input data using mean and scale arrays
void scale_input(float *x) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        x[i] = (x[i] - mean[i]) / scale[i];
    }
}

// SVM prediction function
int predict(float *x) {
    float score = bias[0];
    for (int i = 0; i < NUM_FEATURES; i++) {
        score += weights[0][i] * x[i];
    }
    if (score <= 0)
        return 0;
    return 1;
}

// Function to print a floating-point value with 2 decimal places
void print_float(float val) {
    int int_part = (int)val;
    int frac_part = (int)((val - int_part) * 100);  // 2 decimal precision
    if (frac_part < 0) frac_part *= -1;
    printf("%d.%02d\n", int_part, frac_part);
}

int main() {
    float input[2] = {20, 19000}; // Example input values (feature 1 and feature 2)
    
    // Preprocessing step: scale input
    scale_input(input);
    
    // Predict using SVM
    int label = predict(input);

    // Print result
    printf("Predicted output: %d\n", label);

    return 0;
}


```

## Microcontroller code

## SVM Inference (Embedded AI in C)

## Project Structure
📁 SVM_Inference_Arduino
│
├── sketch.ino          # Main Arduino program
├── svm_model.h         # Model weights and bias
└── scaler.h            # Mean and scale parameters

## Example Code (sketch.ino)
```
#include <Arduino.h>
#include "svm_model.h"
#include "scaler.h"

void scale_input(float *x) {
  for (int i = 0; i < NUM_FEATURES; i++) {
    x[i] = (x[i] - mean[i]) / scale[i];
  }
}

int predict(float *x) {
  float score = bias[0];
  for (int i = 0; i < NUM_FEATURES; i++) {
    score += weights[0][i] * x[i];
  }
  if (score <= 0)
    return 0;
  return 1;
}

void setup() {
  Serial.begin(115200);

  float input[2] = {20, 19000};
  scale_input(input);
  int label = predict(input);

  Serial.print("Predicted output: ");
  Serial.println(label);
}

void loop() {}

```

## Example Model Files
svm_model.h
```
#ifndef SVM_MODEL_H
#define SVM_MODEL_H

#define NUM_FEATURES 2
const float weights[1][NUM_FEATURES] = { {0.001f, 0.0002f} };
const float bias[1] = {-0.85f};

#endif

```

scaler.h

```
#ifndef SCALER_H
#define SCALER_H

#define NUM_FEATURES 2
const float mean[NUM_FEATURES]  = {35.0f, 75000.0f};
const float scale[NUM_FEATURES] = {10.0f, 20000.0f};

#endif

```

## Highlights

* Runs trained ML models without Python or TensorFlow
*  Suitable for SiFive RISC-V, ARM Cortex-M, and Arduino-class boards
* Demonstrates hardware-aware AI deployment
* Works fully inside the Wokwi simulator

## Exanple Output

<img width="1912" height="947" alt="image" src="https://github.com/user-attachments/assets/5bf8f943-f57e-4004-a546-bed0d66a16b1" />

---

# Handwritten Digit Recognition with SVM - From MNIST to Embedded Boards

This project demonstrates handwritten digit classification using the MNIST dataset with a Linear Support Vector Machine (SVM).  
It includes complete preprocessing, training, evaluation, and visualization steps — including confusion matrix and misclassified image analysis.

---

## 📘 Overview

- **Dataset:** MNIST (28×28 grayscale handwritten digits)
- **Model Used:** Linear Support Vector Classifier (LinearSVC)
- **Scaling Method:** StandardScaler (mean normalization and standard deviation scaling)
- **Evaluation Metrics:** Accuracy, Classification Report, Confusion Matrix
- **Visualization:** Matplotlib and Seaborn

---

## ⚙️ Workflow

### 1️⃣ Import Required Libraries
Uses TensorFlow (for MNIST dataset), scikit-learn (for model & metrics), NumPy, Pandas, Matplotlib, and Seaborn.

---

### 2️⃣ Load the MNIST Dataset
The dataset is split into training and test sets:

- **Training data:** 60,000 images  
- **Test data:** 10,000 images

---

### 3️⃣ Reshape and Normalize Data
Each image (28×28) is flattened into a 784-dimensional vector and converted to float32 for processing.

---

### 4️⃣ Feature Scaling
Standardization ensures faster convergence and consistent margin calculations by centering data around mean = 0 and std = 1.

---

### 5️⃣ Model Training
The LinearSVC model is trained with:

- `dual = False` for efficiency  
- `max_iter = 10000` to ensure convergence

---

### 6️⃣ Model Evaluation
Accuracy and classification metrics are computed using:

- `accuracy_score()`  
- `classification_report()`

---

### 7️⃣ Visualization

- Confusion matrix heatmap for class-wise performance  
- Display of a few test images with their true and predicted labels  
- Misclassified image visualization for error analysis

---

**End Result:**  
A simple, interpretable, and efficient linear SVM model achieving ~93–97% accuracy on MNIST test data.

---

# Running MNIST Digit Recognition on the RISC-V Board

## Bare Metal Code
```
#include <stdio.h>
#include <math.h>
#include "svm_model1.h"
#include "scaler1.h"
#include "test_images.h"

// Scale input features using the mean and scale arrays
void scale_input(float *x) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        x[i] = (x[i] - mean[i]) / scale[i];
    }
}

// SVM prediction for a single input vector
int svm_predict(float *x) {
    int best_class = 0;
    float max_score = -INFINITY;

    for (int c = 0; c < NUM_CLASSES; c++) {
        float score = bias[c];
        for (int i = 0; i < NUM_FEATURES; i++) {
            score += weights[c][i] * x[i];
        }
        if (score > max_score) {
            max_score = score;
            best_class = c;
        }
    }
    return best_class;
}

// Utility function to print float values
void print_float(float val) {
    int int_part = (int)val;
    int frac_part = (int)((val - int_part) * 100);  // 2 decimal precision
    if (frac_part < 0) frac_part *= -1;
    printf("%d.%02d\n", int_part, frac_part);
}

// Main function to test multiple images
int main() {
    for (int i = 0; i < NUM_TEST_IMAGES; i++) {
        float input[NUM_FEATURES];
        for (int j = 0; j < NUM_FEATURES; j++) {
            input[j] = test_images[i][j];
        }

        // Preprocess input
        scale_input(input);

        // Predict using SVM
        int predicted = svm_predict(input);
        int actual = test_labels[i];

        // Display result
        printf("Image %d: Predicted = %d, Actual = %d\n", i, predicted, actual);
    }

    return 0;
}

```

## 🧩 Notes

Make sure the following header files are in the same directory as your main C file:

- `svm_model1.h` → contains trained **SVM weights and biases**
- `scaler1.h` → contains **mean** and **scale** arrays for feature normalization
- `test_images.h` → contains **sample test data and corresponding labels**

---

### ⚙️ How It Works

This code performs inference using a pre-trained **Support Vector Machine (SVM)** model exported from Python to C.

1. **Input Scaling:**  
   Each test image vector is normalized using the mean and scale values stored in `scaler1.h`.

2. **Prediction:**  
   The scaled input is passed to the `svm_predict()` function, which computes:

---

score = bias[c] + Σ(weights[c][i] × x[i])

The class with the highest score is chosen as the predicted output.

3. **Testing:**  
The program loops through all test samples defined in `test_images.h`, printing the predicted and actual labels for comparison.

---

### 🖥️ Example Output
Image 0: Predicted = 2, Actual = 2 <br>
Image 1: Predicted = 7, Actual = 7 <br>
Image 2: Predicted = 1, Actual = 1 <br>
...


---

### 📂 Directory Structure
├── main.c <br>
├── svm_model1.h # Contains SVM weights & biases <br>
├── scaler1.h # Contains scaling parameters <br>
├── test_images.h # Contains test dataset samples


---

**This code will loop through all test images, scale them, predict using your trained SVM model, and print each prediction vs the ground truth.**

---

# STM32 Nucleo codes

```
#include <Arduino.h>
#include <math.h>

// ---------- MINI DEMO CONFIG (small to fit RAM/FLASH) ----------
#define NUM_CLASSES     2
#define NUM_FEATURES    16   // tiny feature count (e.g., downsampled or first 16 pixels)
#define NUM_TEST_IMAGES 4    // just a few samples

// Tiny example weights/bias (replace with small real values if you want)
static const float weights[NUM_CLASSES][NUM_FEATURES] = {
  // class 0
  {  0.09f, -0.04f, 0.01f,  0.05f,  -0.02f, 0.03f,  0.01f, -0.01f,
     0.02f,  0.04f, -0.03f, 0.00f,   0.01f, 0.02f, -0.02f,  0.01f },
  // class 1
  { -0.05f,  0.06f,-0.02f, -0.03f,   0.01f,-0.01f,  0.02f,  0.03f,
    -0.02f, -0.04f, 0.04f,  0.01f,  -0.01f, 0.00f,  0.03f, -0.02f }
};
static const float bias[NUM_CLASSES] = { 0.01f, -0.02f };

// Tiny scaler (mean/scale). Keep non-zero scale.
static const float mean[NUM_FEATURES]  = {
  0.50f, 0.50f, 0.50f, 0.50f, 0.50f, 0.50f, 0.50f, 0.50f,
  0.50f, 0.50f, 0.50f, 0.50f, 0.50f, 0.50f, 0.50f, 0.50f
};
static const float scale[NUM_FEATURES] = {
  0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f,
  0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f
};

// Tiny test set (4 samples × 16 features)
static const float test_images[NUM_TEST_IMAGES][NUM_FEATURES] = {
  {0.60f,0.40f,0.55f,0.52f, 0.49f,0.51f,0.47f,0.58f, 0.62f,0.44f,0.48f,0.56f, 0.54f,0.50f,0.53f,0.45f},
  {0.30f,0.52f,0.28f,0.33f, 0.55f,0.57f,0.51f,0.49f, 0.36f,0.40f,0.59f,0.61f, 0.50f,0.47f,0.39f,0.42f},
  {0.72f,0.60f,0.70f,0.68f, 0.63f,0.65f,0.66f,0.71f, 0.69f,0.58f,0.64f,0.62f, 0.67f,0.73f,0.74f,0.61f},
  {0.18f,0.22f,0.25f,0.29f, 0.31f,0.27f,0.24f,0.21f, 0.19f,0.23f,0.33f,0.35f, 0.28f,0.26f,0.20f,0.30f}
};
static const int test_labels[NUM_TEST_IMAGES] = {0, 1, 0, 1};

// ---------- Inference (no extra buffers) ----------
static int svm_predict_scaled(const float *x_raw) {
  int best_class = 0;
  float max_score = -INFINITY;

  for (int c = 0; c < NUM_CLASSES; c++) {
    float score = bias[c];
    for (int i = 0; i < NUM_FEATURES; i++) {
      float xi = (x_raw[i] - mean[i]) / scale[i];
      score += weights[c][i] * xi;
    }
    if (score > max_score) {
      max_score = score;
      best_class = c;
    }
  }
  return best_class;
}

void setup() {
  Serial.begin(115200);
  delay(100);

  int correct = 0;
  for (int n = 0; n < NUM_TEST_IMAGES; n++) {
    const float *x = &test_images[n][0];
    const int y = test_labels[n];
    int pred = svm_predict_scaled(x);
    if (pred == y) correct++;

    Serial.print("Image "); Serial.print(n);
    Serial.print(": Predicted="); Serial.print(pred);
    Serial.print("  Actual="); Serial.println(y);
  }

  float acc = (NUM_TEST_IMAGES > 0) ? (100.0f * correct / NUM_TEST_IMAGES) : 0.0f;
  Serial.print("Accuracy: "); Serial.print(acc, 2); Serial.println("%");
}

void loop() {}

```

## Expected Output:

<img width="1917" height="941" alt="image" src="https://github.com/user-attachments/assets/d4ec1092-e668-4f6e-895d-f82146986ef4" />

It’s a tiny self-contained example that shows how a Support Vector Machine (SVM) classifier makes predictions on microcontrollers with limited RAM/flash.
Everything (weights, bias, test samples) is hard-coded so it can run standalone.

---

# Quantization Demystified - Fitting AI Models on Tiny Devices

## Understanding ML Model Quantization Concepts

Quantization reduces model precision from **32-bit floating point** to **lower bit representations**.

### Key Benefits
- Converts floating-point weights and activations to integer formats  
- Reduces memory footprint and computational complexity significantly  
- Maintains model accuracy while enabling efficient MCU deployment  

### Quantization Approaches
Multiple quantization approaches exist for different deployment scenarios and accuracy requirements:

- **Post-training quantization** — for pre-trained model optimization  
- **Quantization-aware training (QAT)** — for accuracy preservation during training  
- **Dynamic quantization** — for runtime adaptive precision control  

---

# MCU Constraints

## Resource Overview
Microcontrollers present unique challenges with limited memory, processing power, and energy constraints requiring specialized optimization.

### 01. Memory Limitations
MCUs typically have 32KB–1MB RAM and limited flash storage, requiring aggressive model compression techniques.

### 02. Processing Power
Limited computational resources demand efficient algorithms and optimized operations.

---

# Quantization Techniques

## 1. 8-bit Integer Quantization
Standard approach converting 32-bit floats to 8-bit integers with significant memory reduction and computational speedup.

## 2. Mixed-Precision Quantization
Adaptive bit-width allocation using different precisions for layers to optimize accuracy-efficiency trade-offs effectively.

## 3. Sub-byte Quantization
Ultra-low precision using 2–4 bits per parameter for extreme compression enabling deployment on severely constrained devices.

---


# Implementation Challenges

## Key Obstacles

### Accuracy Loss
Quantization introduces numerical errors that can degrade model performance, requiring careful calibration and validation to maintain acceptable inference accuracy.

### Hardware Limitations
MCU instruction sets lack native support for mixed-precision operations, requiring specialized packing algorithms and optimization techniques.

---

# Post-Training Quantization - From 68KB Overflow to MCU-Ready AI

## Quantized SVM Inference on MCU

This program performs **Support Vector Machine (SVM)** inference using quantized model parameters and scaler values stored in C header files. It demonstrates how machine learning inference can be executed efficiently on low-memory embedded systems such as microcontrollers.

### Overview

The model parameters (weights, bias, mean, and scale) are pre-quantized into **8-bit integers (int8_t)** for memory and computation efficiency. The code performs scaling, prediction, and evaluation on test samples directly on the MCU.

### Files Included

- **svm_model_q.h** – Contains quantized SVM weights, biases, and the scaling factor (`weight_scale`, `bias_scale`).
- **scaler_q.h** – Holds quantized feature scaling parameters (`mean`, `scale`) and their corresponding scale constants (`mean_scale`, `scale_scale`).
- **test_images_q.h** – Stores test feature vectors and their labels.

### Functions

#### `void scale_input(int8_t *x)`
Normalizes each feature using quantized mean and scale values:
```c
x[i] = (x[i] - (mean[i] * mean_scale)) / (scale[i] * scale_scale);
```

```
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "svm_model_q.h"
#include "scaler_q.h"
#include "test_images_q.h"

void scale_input(int8_t *x) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        x[i] = (x[i] - (mean[i] * mean_scale)) / (scale[i] * scale_scale);
    }
}

int predict(int8_t *x) {
    int best_class = 0;
    float max_score = -INFINITY;
    for (int c = 0; c < NUM_CLASSES; ++c) {
        float score = bias[c] * bias_scale;
        for (int i = 0; i < NUM_FEATURES; i++) {
            score += weights[c][i] * x[i] * weight_scale;
        }
        if (score > max_score) {
            max_score = score;
            best_class = c;
        }
    }
    return best_class;
}


void print_float(float val) {
    int int_part = (int)val;
    int frac_part = (int)((val - int_part) * 100);  // 2 decimal places
    if (frac_part < 0) frac_part *= -1;
    printf("%d.%02d\n", int_part, frac_part);
}

int main() {
    for (int i = 0; i < NUM_TEST_IMAGES; i++) {
        scale_input(test_images[i]);
        int predicted = predict(test_images[i]);
        int actual = test_labels[i];
        printf("Image %d: Predicted = %d, Actual = %d\n", i, predicted, actual);
    }
    return 0;
}
```
---

# SVM Inference on ESP32 (Quantized Model)

This project demonstrates running a **quantized Support Vector Machine (SVM)** model on the **ESP32 microcontroller** using C.  
The code performs feature scaling, model inference, and prediction display through the serial monitor — ideal for **Edge AI and embedded ML** experiments.

---

## Features
- Runs quantized SVM inference directly on the ESP32.
- Uses fixed-point scaled parameters (`svm_model_q.h`, `scaler_q.h`, `test_images_q.h`).
- Implements lightweight preprocessing (`scale_input()`).
- Fully compatible with both **Arduino** and **ESP-IDF** frameworks.
- Demonstrates end-to-end workflow: scaling → inference → result printout.

---

## 🧩 File Structure
├── main.c / svm_esp32.ino # Main inference code <br>
├── svm_model_q.h # Model weights, biases, and scaling constants <br>
├── scaler_q.h # Mean and scale arrays for normalization <br>
├── test_images_q.h # Test samples and expected labels <br>
├── README.md # Project documentation <br>

---


---

## How to Run

###  Arduino IDE
1. Install **ESP32 Boards** via Arduino Board Manager.
2. Create a new sketch (e.g., `svm_esp32.ino`).
3. Copy the code from `main.c` into the sketch.
4. Place your header files in the same directory.
5. Select your ESP32 board (e.g., *ESP32 Dev Module*).
6. Upload and open the Serial Monitor (115200 baud).

## Main C Code
```
#include <Arduino.h>
#include <stdint.h>
#include <math.h>

#include "svm_model_q.h"
#include "scaler_q.h"
#include "test_images_q.h"

// Convert float to int8_t with saturation
static inline int8_t f2i8(float v) {
  if (v > 127.0f) v = 127.0f;
  if (v < -128.0f) v = -128.0f;
  return (int8_t)lrintf(v);
}

void scale_input(int8_t *x) {
  for (int i = 0; i < NUM_FEATURES; i++) {
    // Do math in float to avoid integer truncation
    float xf = (float)x[i];
    float m  = mean[i]  * mean_scale;
    float s  = scale[i] * scale_scale;
    float z  = (xf - m) / (s == 0.0f ? 1.0f : s);
    x[i] = f2i8(z);
  }
}

int predict(const int8_t *x) {
  int   best_class = 0;
  float max_score  = -INFINITY;
  for (int c = 0; c < NUM_CLASSES; ++c) {
    float score = bias[c] * bias_scale;
    for (int i = 0; i < NUM_FEATURES; i++) {
      score += (weights[c][i] * weight_scale) * (float)x[i];
    }
    if (score > max_score) {
      max_score  = score;
      best_class = c;
    }
  }
  return best_class;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { /* wait USB CDC if needed */ }

  for (int i = 0; i < NUM_TEST_IMAGES; i++) {
    scale_input(test_images[i]);
    int predicted = predict(test_images[i]);
    int actual    = test_labels[i];
    Serial.printf("Image %d: Predicted = %d, Actual = %d\r\n", i, predicted, actual);
  }
}

void loop() {
  // no-op
}

```

You should see:
<img width="1913" height="941" alt="image" src="https://github.com/user-attachments/assets/1878a750-88b5-49d8-8890-ec34f1073ad9" />

---

# The Brain’s Building Blocks: Biological Neurons

- The human brain contains approximately **86 billion neurons**, interconnected by trillions of synapses, forming an incredibly complex network.  

- Neurons communicate through a sophisticated interplay of **electrical impulses** (action potentials) and **chemical signals**.  

- Key features include:  
  - **Dendrites** that receive input  
  - **Axons** that send output  
  - **Synapses** that modulate signal strength, enabling dynamic information flow  

<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/94122d82-e6e6-4d74-a49e-31e114841c3a" />

Reference: [View on Tinker](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.70009&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

---

# ⚙️ Common Activation Functions in Neural Networks

Activation functions introduce **non-linearity** into neural networks, enabling them to learn complex data patterns and relationships.

---

## 1. Tanh (Hyperbolic Tangent)

**Equation:**  
tanh(x)

**Description:**  
- Range: (-1, 1)  
- Output is zero-centered  
- Can saturate for large |x| values  

---

## 2. ReLU (Rectified Linear Unit)

**Equation:**  
f(x) = max(0, x)

**Description:**  
- Range: [0, ∞)  
- Simple and efficient to compute  
- Helps mitigate vanishing gradients  
- May lead to “dead neurons” for negative inputs  

---

## 3. Sigmoid

**Equation:**  
σ(x) = 1 / (1 + e^(-x))

**Description:**  
- Range: (0, 1)  
- Smooth and differentiable  
- Commonly used for binary classification  
- Can saturate and cause slow convergence for large |x|  

---

## 4. Linear (Identity Function)

**Equation:**  
f(x) = x

**Description:**  
- Range: (-∞, ∞)  
- No non-linearity  
- Used in regression models or as an output layer  

---

### Visualization

Below is a comparison of these activation functions:

<img width="789" height="529" alt="image" src="https://github.com/user-attachments/assets/94aec4b7-35e0-45df-b0f0-99204c5b2fb4" />

---

# Deep Neural Networks

# Visual ANN — Interactive Neural Network Visualization

This project provides an **interactive visualization of an Artificial Neural Network (ANN)** that demonstrates how data flows through layers, neurons, and connections in real time.

---

## Overview

- The **left panel** represents the **input grid**, where each square acts as a pixel or feature input.  
  Users can draw directly on this grid, simulating how an ANN receives raw input data.  

- The **middle section** visualizes the **hidden layers** of the network:  
  - **Layer 1** (pink nodes) receives the input signals.  
  - **Layer 2** (orange nodes) processes those signals further through weighted connections.  
  - Each connection’s color intensity indicates the weight’s magnitude and sign — positive or negative influence on the next layer.

- The **right panel** displays the **output neurons**, labeled from **0 to 9**, corresponding to the network’s possible classification outputs (e.g., handwritten digits).  
  - The highlighted green box shows the predicted class.  
  - The **“Prediction: N/A”** label updates dynamically once input is processed.

---

## Features

- **Real-time forward propagation visualization** — watch activations and weights update as data flows through the network.  
- **Interactive input grid** — users can draw custom patterns to see how the model interprets them.  
- **Dynamic color-coded connections** — helps understand how neurons activate and contribute to the final prediction.  
- **Lightweight and educational** — ideal for learning neural network concepts interactively.

---

## Example Visualization

<img width="584" height="421" alt="image" src="https://github.com/user-attachments/assets/041858eb-8f70-4612-ba80-bdcbdf20aba0" />


---

## Use Case

This visualization is perfect for:
- Teaching the fundamentals of neural networks  
- Demonstrating **feedforward propagation**  
- Understanding the impact of weights and activations  
- Exploring the relationship between **input**, **hidden**, and **output layers**

---

### Future Extensions
- Add support for **custom architectures** (variable layer sizes).  
- Integrate **activation function visualization** (Sigmoid, ReLU, Tanh).  
- Allow real-time weight updates to visualize **backpropagation**.

---

# Training Bit-Quantized Neural Network Implementation with Quantization-Aware Training

## RISC-V Edge AI 

This repository serves as a starting framework for deploying **quantized AI models** on **RISC-V-based edge devices**.  <br>
The project demonstrates how to **train, quantize, and deploy lightweight neural networks** on microcontrollers using the **VSD Squadron Mini board** with **CH32V003F4U6** and **OV7670 camera**.

---

### Overview
The **RISC-V Edge AI Workshop** showcases a complete end-to-end workflow:

- **Model Training:** Python scripts for dataset training and quantization  
- **Embedded Deployment:** C/C++ inference code optimized for RISC-V boards  
- **Hardware Integration:** Camera interface, real-time prediction, and PCB design examples  
- **Edge Inference:** Demonstrates on-device AI without external compute resources  

---

### Repository Structure
```bash
├── Training/               # Scripts for model training and quantization  
├── Camera_Interfacing/ # Code for camera integration and board setup  
├── Prediction/         # Quantized inference implementation  
├── PCB_DESIGN/             # Schematic and PCB layout files  
└── README.md               # Project documentation
```

🔗 **Original Repository:** [RiscV_Edge_AI_Workshop by Dhanvanti Bhavsar](https://github.com/dhanvantibhavsar/RiscV_Edge_AI_Workshop)

---















