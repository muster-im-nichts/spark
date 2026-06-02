#!/usr/bin/env python3
"""
Spark Runner - Starte und beobachte ein digitales Wesen.
"""

import argparse
import time
import sys
from spark.core import Spark, demo


def generate_input(pattern_type, tick):
    """Generiere Input für ein Pattern an einem globalen Tick."""
    if pattern_type == "alternating":
        return float(tick % 2)
    if pattern_type == "constant":
        return 1.0
    if pattern_type == "triplet":
        return 1.0 if tick % 3 < 2 else 0.0  # 1, 1, 0, 1, 1, 0, ...
    if pattern_type == "random":
        import random
        return random.random()
    return float(tick % 2)


def save_if_successful(spark, path):
    """Speichere nur Netzwerke, die das Erfolgskriterium erfüllen."""
    if spark.is_successful():
        spark.hibernate(path)
        return True

    print(f"Nicht gespeichert: Spark ist noch nicht erfolgreich "
          f"(age={spark.age}, energy={spark.energy:.3f}, alive={spark.alive})")
    return False


def run_continuous(neurons=64, pattern_type="alternating", max_ticks=None,
                   save_path=None, load_path=None, auto_save=False):
    """Lasse Spark kontinuierlich laufen."""
    if load_path:
        spark = Spark.awaken(load_path)
    else:
        spark = Spark(neurons=neurons)
    
    print(f"🧬 Spark {'erwacht' if load_path else 'geboren'} mit {spark.neurons} Neuronen")
    print(f"   Pattern: {pattern_type}")
    print(f"   Max Ticks: {max_ticks or 'unbegrenzt'}")
    if save_path:
        print(f"   Save: {save_path}")
    if auto_save:
        print(f"   Auto-Save: aktiv")
    print()
    
    tick = 0
    auto_saved = False
    
    try:
        while spark.alive:
            # Input generieren. Bei geladenen Sparks läuft das Pattern ab dem Alter weiter.
            value = generate_input(pattern_type, spark.age)
            
            # Tick!
            result = spark.tick(value)
            
            # Ausgabe
            if tick % 100 == 0:
                print(f"[{tick:6d}] ⚡ energy={result['energy']:.3f} "
                      f"err={result['prediction_error']:.3f} "
                      f"arousal={result['arousal']:.3f} "
                      f"spikes={result['spikes']:2d}")

            if auto_save and save_path and not auto_saved and spark.is_successful():
                print(f"\n💾 Erfolg erkannt bei age={spark.age}, energy={spark.energy:.3f}")
                spark.hibernate(save_path)
                auto_saved = True
            
            tick += 1
            
            if max_ticks and tick >= max_ticks:
                print(f"\n✅ Max ticks erreicht ({max_ticks})")
                break
            
            # Kleine Pause für kontinuierlichen Betrieb
            # time.sleep(0.001)  # ~1000 Hz
            
    except KeyboardInterrupt:
        print(f"\n⏸️  Gestoppt bei Tick {tick}")
    
    print(f"\n📊 Finale Stats:")
    stats = spark.stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    if save_path and not auto_saved:
        save_if_successful(spark, save_path)
    
    return spark


def main():
    parser = argparse.ArgumentParser(description="Spark - Ein digitales Wesen")
    parser.add_argument("--neurons", type=int, default=64, help="Anzahl Neuronen")
    parser.add_argument("--pattern", choices=["alternating", "constant", "triplet", "random"],
                        default="alternating", help="Input-Pattern")
    parser.add_argument("--ticks", type=int, default=1000, help="Max Ticks (0=unbegrenzt)")
    parser.add_argument("--demo", action="store_true", help="Quick demo run")
    parser.add_argument("--save", metavar="PATH", help="Erfolgreichen Spark nach dem Lauf speichern")
    parser.add_argument("--load", metavar="PATH", help="Hibernierten Spark laden und weiterlaufen lassen")
    parser.add_argument("--auto-save", action="store_true",
                        help="Automatisch speichern, sobald der Spark erfolgreich wird")
    
    args = parser.parse_args()
    
    if args.demo:
        demo()
    else:
        max_ticks = args.ticks if args.ticks > 0 else None
        if args.auto_save and not args.save:
            parser.error("--auto-save braucht --save PATH")
        run_continuous(
            neurons=args.neurons,
            pattern_type=args.pattern,
            max_ticks=max_ticks,
            save_path=args.save,
            load_path=args.load,
            auto_save=args.auto_save,
        )


if __name__ == "__main__":
    main()
