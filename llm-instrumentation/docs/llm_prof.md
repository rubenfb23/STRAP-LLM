# llm-prof eBPF CLI

`llm-prof` es una utilidad de línea de comandos que orquesta colecciones eBPF de
bajo overhead para pipelines de inferencia. Permite capturar métricas de disco,
syscalls, memoria, scheduler y red por intervalos fijos.

## Instalación

```bash
pip install .[ebpf]
```

## Uso básico

```bash
sudo llm-prof run --pid <PID_OBJETIVO> --interval 1.0 --output profiler.jsonl
```

Flags relevantes:

- `--only`: elige dominios (`disk,syscalls,sync,mem,sched,net`).
- `--sample-read-write`: muestreo 1:N para syscalls intensivas.
- `--labels`: etiquetas adicionales `key=value` adjuntas a cada evento.
- `--disable-correlation`: desactiva la propagación de `corr_id`.

Los snapshots quedan en JSONL con estructura `ts_ns`, `type`, `pid`, `fields...`.

## Correlación de solicitudes

Puedes propagar IDs de correlación desde Python:

```python
from llm_instrumentation.ebpf.correlation import CorrelationContext

corr = CorrelationContext.default()

def handle_request(req):
    corr_id = corr.new_request()
    # ... trabajo ...
    corr.set(None)
```

Las métricas de los collectors adjuntarán `corr_id` si está activo.
