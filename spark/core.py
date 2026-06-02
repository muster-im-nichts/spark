"""
Spark - Ein digitales Wesen.

Spiking Neural Network mit:
- STDP (Spike-Timing-Dependent Plasticity) für lokales Lernen
- Prediction-Error als Drive (Surprise Minimization)
- Wachstum und Hibernation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import json
import time


@dataclass
class SparkState:
    """Serialisierbarer Zustand für Hibernation."""
    neurons: int
    membrane: np.ndarray
    threshold: np.ndarray
    weights: np.ndarray
    energy: float
    curiosity: float
    arousal: float
    age: int  # Ticks seit Geburt
    prediction_history: list = field(default_factory=list)
    last_prediction: Optional[float] = None
    alive: bool = True
    
    def save(self, path: str):
        """Speichere Zustand für Hibernation."""
        np.savez_compressed(
            path,
            membrane=self.membrane,
            threshold=self.threshold,
            weights=self.weights,
            prediction_history=np.array(self.prediction_history, dtype=float),
            meta=np.array([
                self.energy,
                self.curiosity,
                self.arousal,
                self.age,
                self.neurons,
                np.nan if self.last_prediction is None else self.last_prediction,
                1.0 if self.alive else 0.0,
            ])
        )
    
    @classmethod
    def load(cls, path: str) -> "SparkState":
        """Lade Zustand aus Hibernation."""
        data = np.load(path)
        meta = data['meta']
        return cls(
            neurons=int(meta[4]),
            membrane=data['membrane'],
            threshold=data['threshold'],
            weights=data['weights'],
            energy=float(meta[0]),
            curiosity=float(meta[1]),
            arousal=float(meta[2]),
            age=int(meta[3]),
            prediction_history=data['prediction_history'].tolist() if 'prediction_history' in data else [],
            last_prediction=None if len(meta) < 6 or np.isnan(meta[5]) else float(meta[5]),
            alive=bool(meta[6]) if len(meta) >= 7 else True,
        )


class Spark:
    """
    Ein digitales Wesen basierend auf Spiking Neural Networks.
    
    Drive: Vorhersage. Das Netz versucht den nächsten Input vorherzusagen.
    Überraschung = Energie-Verlust. Korrekte Vorhersage = Energie-Gewinn.
    """
    
    # Konstanten
    SPIKE_THRESHOLD = 1.0
    MEMBRANE_DECAY = 0.9  # Wie schnell vergisst ein Neuron?
    STDP_WINDOW = 20  # ms für STDP
    LEARN_RATE = 0.01
    METABOLISM_COST = 0.001  # Energie pro Tick
    PREDICTION_REWARD = 0.01
    PREDICTION_PENALTY = 0.005
    SUCCESS_MIN_AGE = 500
    SUCCESS_MIN_ENERGY = 1.0
    
    def __init__(self, neurons: int = 64, seed: Optional[int] = None):
        """
        Initialisiere ein neues Spark.
        
        Args:
            neurons: Anzahl der Neuronen
            seed: Random seed für Reproduzierbarkeit
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.neurons = neurons
        
        # Neuronaler Zustand
        self.membrane = np.zeros(neurons)  # Membranpotential
        self.threshold = np.ones(neurons) * self.SPIKE_THRESHOLD
        
        # Verbindungen (sparse-ish, ~10% Konnektivität)
        self.weights = np.random.randn(neurons, neurons) * 0.1
        self.weights *= (np.random.rand(neurons, neurons) < 0.1)  # Sparsity
        np.fill_diagonal(self.weights, 0)  # Keine Selbst-Verbindungen
        
        # Interner Zustand - "Gefühle"
        self.energy = 1.0
        self.curiosity = 0.5
        self.arousal = 0.3
        
        # Tracking
        self.age = 0
        self.spike_history = []  # Letzte N Spike-Zeiten pro Neuron
        self.prediction_error_history = []
        
        # Input/Output Neuronen (erste 8 = Input, letzte 8 = Output/Prediction)
        self.input_neurons = list(range(8))
        self.output_neurons = list(range(neurons - 8, neurons))
        
        # Letzter Input für Prediction-Vergleich
        self.last_prediction = None
        self.alive = True
    
    def _inject_input(self, value: float) -> np.ndarray:
        """Injiziere Input in die Input-Neuronen."""
        input_signal = np.zeros(self.neurons)
        # Verteile den Input-Wert auf die Input-Neuronen
        for i, idx in enumerate(self.input_neurons):
            # Etwas Variation pro Neuron
            input_signal[idx] = value * (0.8 + 0.4 * (i / len(self.input_neurons)))
        return input_signal
    
    def _propagate(self, external_input: np.ndarray) -> np.ndarray:
        """
        Propagiere Spikes durch das Netz.
        
        Returns:
            Array mit Spike-Zeiten (0 = kein Spike, >0 = Spike-Zeit)
        """
        # Membrane Decay
        self.membrane *= self.MEMBRANE_DECAY
        
        # Externe Inputs addieren
        self.membrane += external_input
        
        # Interne Aktivierung von vorherigen Spikes
        if self.spike_history:
            recent_spikes = self.spike_history[-1]
            internal_input = self.weights.T @ recent_spikes
            self.membrane += internal_input * 0.5
        
        # Spikes generieren
        spikes = np.zeros(self.neurons)
        firing = self.membrane >= self.threshold
        spikes[firing] = self.age  # Spike-Zeit = aktuelles Alter
        
        # Reset nach Spike
        self.membrane[firing] = 0
        
        # Spike-History updaten (FIFO, max 100)
        self.spike_history.append(spikes.copy())
        if len(self.spike_history) > 100:
            self.spike_history.pop(0)
        
        return spikes
    
    def _learn_stdp(self, spikes: np.ndarray):
        """
        STDP: Spike-Timing-Dependent Plasticity.
        
        Neurons that fire together wire together.
        Aber: Timing matters!
        - Pre vor Post = Verstärkung (LTP)
        - Post vor Pre = Abschwächung (LTD)
        """
        if len(self.spike_history) < 2:
            return
        
        current = spikes
        previous = self.spike_history[-2]
        
        for i in range(self.neurons):
            for j in range(self.neurons):
                if i == j:
                    continue
                
                # Beide müssen gefeuert haben
                if current[i] > 0 and previous[j] > 0:
                    # j feuerte vor i -> Verstärkung
                    dt = current[i] - previous[j]
                    if 0 < dt < self.STDP_WINDOW:
                        self.weights[j, i] += self.LEARN_RATE * np.exp(-dt / self.STDP_WINDOW)
                
                if previous[i] > 0 and current[j] > 0:
                    # i feuerte vor j -> Abschwächung
                    dt = current[j] - previous[i]
                    if 0 < dt < self.STDP_WINDOW:
                        self.weights[i, j] -= self.LEARN_RATE * 0.5 * np.exp(-dt / self.STDP_WINDOW)
        
        # Weights clippen
        self.weights = np.clip(self.weights, -1.0, 1.0)
    
    def _get_prediction(self, spikes: np.ndarray) -> float:
        """Extrahiere Vorhersage aus Output-Neuronen."""
        output_activity = spikes[self.output_neurons]
        # Normalisiere auf 0-1
        if np.any(output_activity > 0):
            return np.mean(output_activity > 0)
        return 0.5  # Default: unsicher
    
    def _update_state(self, prediction_error: float):
        """Update interner Zustand basierend auf Prediction Error."""
        # Energie
        if prediction_error < 0.3:
            self.energy += self.PREDICTION_REWARD
        else:
            self.energy -= self.PREDICTION_PENALTY * prediction_error
        
        self.energy = np.clip(self.energy, 0, 2.0)
        
        # Arousal steigt bei Überraschung
        self.arousal = 0.7 * self.arousal + 0.3 * prediction_error
        
        # Curiosity sinkt wenn alles vorhersagbar ist
        self.curiosity = 0.9 * self.curiosity + 0.1 * (1 - prediction_error)
        
        # Metabolism
        self.energy -= self.METABOLISM_COST
        
        # Tod?
        if self.energy <= 0:
            self.alive = False
    
    def tick(self, input_value: float) -> dict:
        """
        Ein Lebens-Tick.
        
        Args:
            input_value: Der aktuelle Input (0-1)
            
        Returns:
            Dict mit Diagnose-Infos
        """
        if not self.alive:
            return {"alive": False, "cause": "energy_depleted"}
        
        self.age += 1
        
        # 1. Input injizieren
        external = self._inject_input(input_value)
        
        # 2. Spikes propagieren
        spikes = self._propagate(external)
        
        # 3. Vorhersage extrahieren
        prediction = self._get_prediction(spikes)
        
        # 4. Prediction Error berechnen (wenn wir einen vorherigen Input haben)
        prediction_error = 0.5  # Default
        if self.last_prediction is not None:
            prediction_error = abs(input_value - self.last_prediction)
            self.prediction_error_history.append(prediction_error)
            if len(self.prediction_error_history) > 1000:
                self.prediction_error_history.pop(0)
        
        # 5. Lernen (STDP)
        self._learn_stdp(spikes)
        
        # 6. Zustand updaten
        self._update_state(prediction_error)
        
        # 7. Prediction für nächsten Tick speichern
        self.last_prediction = prediction
        
        return {
            "alive": self.alive,
            "age": self.age,
            "energy": round(self.energy, 4),
            "arousal": round(self.arousal, 4),
            "curiosity": round(self.curiosity, 4),
            "prediction": round(prediction, 4),
            "prediction_error": round(prediction_error, 4),
            "spikes": int(np.sum(spikes > 0)),
            "mean_weight": round(np.mean(np.abs(self.weights)), 6)
        }
    
    def get_state(self) -> SparkState:
        """Exportiere aktuellen Zustand."""
        return SparkState(
            neurons=self.neurons,
            membrane=self.membrane.copy(),
            threshold=self.threshold.copy(),
            weights=self.weights.copy(),
            energy=self.energy,
            curiosity=self.curiosity,
            arousal=self.arousal,
            age=self.age,
            prediction_history=self.prediction_error_history.copy(),
            last_prediction=self.last_prediction,
            alive=self.alive,
        )
    
    def hibernate(self, path: str):
        """Speichere Zustand und 'schlafe ein'."""
        self.get_state().save(path)
        print(f"Spark hibernated to {path} (age={self.age}, energy={self.energy:.3f})")
    
    @classmethod
    def awaken(cls, path: str) -> "Spark":
        """Erwecke aus Hibernation."""
        state = SparkState.load(path)
        spark = cls(neurons=state.neurons)
        spark.membrane = state.membrane
        spark.threshold = state.threshold
        spark.weights = state.weights
        spark.energy = state.energy
        spark.curiosity = state.curiosity
        spark.arousal = state.arousal
        spark.age = state.age
        spark.prediction_error_history = state.prediction_history
        spark.last_prediction = state.last_prediction
        spark.alive = state.alive
        print(f"Spark awakened from {path} (age={spark.age}, energy={spark.energy:.3f})")
        return spark

    def is_successful(self) -> bool:
        """Ein Spark gilt als erfolgreich, wenn er alt genug ist und Energie hält."""
        return self.alive and self.age >= self.SUCCESS_MIN_AGE and self.energy >= self.SUCCESS_MIN_ENERGY
    
    def stats(self) -> dict:
        """Statistiken über das aktuelle Netz."""
        return {
            "neurons": self.neurons,
            "age": self.age,
            "alive": self.alive,
            "energy": self.energy,
            "connections": int(np.sum(np.abs(self.weights) > 0.01)),
            "mean_prediction_error": np.mean(self.prediction_error_history) if self.prediction_error_history else None,
            "weight_stats": {
                "mean": float(np.mean(self.weights)),
                "std": float(np.std(self.weights)),
                "max": float(np.max(self.weights)),
                "min": float(np.min(self.weights))
            }
        }


# Convenience für schnelle Tests
def demo():
    """Schneller Demo-Run."""
    spark = Spark(neurons=64, seed=42)
    
    # Einfaches Pattern: 1, 0, 1, 0, ...
    pattern = [1.0, 0.0] * 500  # 1000 Ticks
    
    print("Starting Spark demo...")
    print(f"Pattern: alternating 1, 0")
    print(f"Initial stats: {spark.stats()}")
    print()
    
    for i, value in enumerate(pattern):
        result = spark.tick(value)
        
        if not result["alive"]:
            print(f"Spark died at tick {i}!")
            break
        
        if i % 100 == 0:
            print(f"Tick {i:4d}: energy={result['energy']:.3f}, "
                  f"pred_err={result['prediction_error']:.3f}, "
                  f"spikes={result['spikes']:2d}")
    
    print()
    print(f"Final stats: {spark.stats()}")
    return spark


if __name__ == "__main__":
    demo()
