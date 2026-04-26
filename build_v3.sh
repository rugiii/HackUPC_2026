#!/bin/bash
# build_v3.sh — Compila warehouse_core_v3.so con -O2 + OpenMP
# Ejecutar antes de usar warehouse_optimizer_v5.py

set -e

echo "Compilando warehouse_core_v3.so con OpenMP..."

gcc -O2 -march=native -ffast-math -fopenmp \
    -shared -fPIC \
    -o warehouse_core_v3.so \
    warehouse_core_v3.c \
    -lm

if [ $? -eq 0 ]; then
    echo "✓ warehouse_core_v3.so compilado correctamente"
    ls -lh warehouse_core_v3.so
    echo ""
    echo "Para controlar el número de threads:"
    echo "  export OMP_NUM_THREADS=8   # usa 8 threads"
    echo "  python warehouse_optimizer_v5.py ..."
else
    echo "✗ Error de compilación"
    exit 1
fi
