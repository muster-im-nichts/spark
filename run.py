#!/usr/bin/env python3
"""
Spark Runner - Starte und beobachte ein digitales Wesen.
"""

import argparse
import time
import sys
from spark.core import Spark, demo


def run_continuous(neurons=64, pattern_type="alternating", max_ticks=None):
    """Lasse Spark kontinuierlich laufen."""
    spark = Spark(neurons=neurons)
    
    print(f"🧬 Spark geboren mit {neurons} Neuronen")
    print(f"   Pattern: {pattern_type}")
    print(f"   Max Ticks: {max_ticks or 'unbegrenzt'}")
    print()
    
    tick = 0
    
    try:
        while spark.alive:
            # Input generieren
            if pattern_type == "alternating":
                value = float(tick % 2)
            elif pattern_type == "constant":
                value = 1.0
            elif pattern_type == "triplet":
                value = 1.0 if tick % 3 < 2 else 0.0  # 1, 1, 0, 1, 1, 0, ...
            elif pattern_type == "random":
                import random
                value = random.random()
            else:
                value = float(tick % 2)
            
            # Tick!
            result = spark.tick(value)
            
            # Ausgabe
            if tick % 100 == 0:
                print(f"[{tick:6d}] ⚡ energy={result['energy']:.3f} "
                      f"err={result['prediction_error']:.3f} "
                      f"arousal={result['arousal']:.3f} "
                      f"spikes={result['spikes']:2d}")
            
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
    
    # Hibernation anbieten
    if spark.alive:
        spark.hibernate("/opt/spark/spark.hibernated.npz")
    
    return spark


def main():
    parser = argparse.ArgumentParser(description="Spark - Ein digitales Wesen")
    parser.add_argument("--neurons", type=int, default=64, help="Anzahl Neuronen")
    parser.add_argument("--pattern", choices=["alternating", "constant", "triplet", "random"],
                        default="alternating", help="Input-Pattern")
    parser.add_argument("--ticks", type=int, default=1000, help="Max Ticks (0=unbegrenzt)")
    parser.add_argument("--demo", action="store_true", help="Quick demo run")
    
    args = parser.parse_args()
    
    if args.demo:
        demo()
    else:
        max_ticks = args.ticks if args.ticks > 0 else None
        run_continuous(args.neurons, args.pattern, max_ticks)


if __name__ == "__main__":
    main()
