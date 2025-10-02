# Benchmarks

Árbol destinado a almacenar resultados y gráficas generadas por distintos sistemas de instrumentación.

```
benchmarks/
└── systems/
    └── I-O/           # PNG generadas por scripts/analyze_tracepoints.py
```

El analizador crea archivos con el patrón `*-latency_hist.png`, `*-queue_depth.png` y `*-requests.png` por dispositivo. Siéntete libre de añadir directorios hermanos (por ejemplo `CPU/`, `Network/`) para otros artefactos de bench.
