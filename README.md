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

## 🚀 What This Project Is Really About

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

### 🧠 Key Insight

Clock speed defines **how many opportunities per second** your processor has to perform work.  
The **real speed** of inference depends on how many cycles each instruction takes and how efficiently the CPU uses each tick.

---

# Best-Fitting Lines 101 - Getting Started With ML

## 📈 Linear Regression Visualization

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
👉 [View on Desmos](https://www.desmos.com/calculator/ylifeiyfqw)

---

# Gradient Descent Unlocked

# 📘 Linear Regression from Scratch using Python

This project demonstrates how to build a **Linear Regression model from scratch** using **NumPy**, **Pandas**, and **Matplotlib** — without using prebuilt machine learning libraries like scikit-learn.

The model predicts students’ exam scores based on the number of hours they studied and explains how **Gradient Descent** works to minimize error and find the best-fit line.

---

## 🎯 Objective

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

## 📂 Dataset

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



















