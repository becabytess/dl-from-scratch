# Autograd Engine & Neural Network from Scratch in C

A complete reverse-mode automatic differentiation engine and feedforward neural network built from scratch in pure C — no libraries, no frameworks.

## Features

- **Computation graph** — dynamic DAG of scalar `Value` nodes linked by arithmetic ops
- **Reverse-mode autodiff** — backpropagation via topological sort and the chain rule
- **Fully flexible feedforward network** — configurable depth and width (default: 50 → 32 → 16 → 8 → 1)
- **MSE loss + SGD** — trains end-to-end on synthetic data

## Build & run

```bash
gcc main.c -o main.exe
./main.exe
```
