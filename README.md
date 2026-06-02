# Spark 🧬

Ein Experiment in digitalem Leben.

**Was ist das?** Ein Spiking Neural Network das kontinuierlich läuft, lokal lernt (STDP), 
wächst wenn nötig, und hiberniert wenn Ressourcen knapp sind.

**Drive:** Vorhersage. Das Netz versucht den nächsten Input vorherzusagen.
Überraschung = Unbehagen. Korrekte Vorhersage = Wohlbefinden.

**Prinzipien:**
- Kein Backprop - nur lokales Hebb'sches Lernen (STDP)
- Kein Training/Inference Split - lernt und lebt gleichzeitig
- Event-driven, nicht turn-based
- Wächst und schrumpft nach Bedarf

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python spark.py
```

## Status

🌱 Embryo - erste Spike-Propagation + STDP + Prediction Error

## Architektur

```
┌─────────────────────────────────────────────┐
│  Spark - Digital Life                       │
│                                             │
│  Input Stream → [Sensory Neurons]           │
│                      ↓                      │
│               [Internal Neurons]            │
│                (Spiking, STDP)              │
│                      ↓                      │
│               [Prediction Neurons]          │
│                      ↓                      │
│         Compare with next Input             │
│                      ↓                      │
│         Prediction Error → Drive            │
│                                             │
│  Loop: ~100 Hz, async, continuous           │
└─────────────────────────────────────────────┘
```

## Lizenz

MIT
