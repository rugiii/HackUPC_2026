# HackUPC_2026
Maximizing warehouse spatial efficiency in seconds using a custom C/Python architecture.

# 📦 Mecalux Warehouse Optimizer | HackUPC 2026

A high-performance optimization engine designed to solve the **Mecalux** challenge at **HackUPC 2026**. This project calculates the optimal bay layout in an arbitrary warehouse, maximizing storage capacity, minimizing costs, and making the most of the available space.

## 🚀 Key Features

* **Advanced Hybrid Algorithm:** Combines *Beam Search* (exploring multiple partial states in parallel) with *Local Search* (post-greedy swap/remove) to escape local minima and find highly optimized layouts.
* **Multiprocessing & OpenMP:** Utilizes 100% of the CPU. Evaluates states in parallel in Python and delegates heavy geometric collision calculations to a C extension parallelized with *OpenMP*.
* **Zero Python Dependencies:** The optimizer (v5) is written using **exclusively the Python Standard Library**. It does not require `pip install`, `numpy`, or `shapely`.
* **Integrated 3D Web Viewer:** A "Plug-and-Play" web tool (HTML/JS) that features a fixed auto-adjusting top-down view, corporate logo projection, and a code inspection panel.

---

## 🛠️ Prerequisites

To run the optimizer at maximum performance, you will need:
* **Python 3.x** (Any recent version).
* **GCC** (To compile the C extension with OpenMP support).

*Note: If the C code cannot be compiled on the target machine, the Python script includes a fallback pure-Python version, though it will be significantly slower.*

---

## ⚙️ Compilation & Installation

No Python packages need to be installed. You only need to compile the mathematical C core to generate the `warehouse_core.so` shared library:

```bash
# On Linux / macOS
gcc -shared -o warehouse_core.so -fPIC -O3 warehouse_core.c -fopenmp
