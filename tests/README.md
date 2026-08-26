# Pipeline V0 tests

The test suite contains both pure/unit tests and live integration tests.

Run the complete suite with:

```text
/home/leiyuxin/miniconda3/envs/ss/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

Pure and unit tests do not require Habitat or Replica assets. Live integration
tests require the `ss` conda environment with Python 3.9, habitat-sim 0.2.2,
the Replica `office_0` scene assets and navmesh, and the CPU acoustic
AudioSensor backend. The live configuration uses quaternion import before
`habitat_sim`, materials OFF, 16 kHz audio, and binaural output.

If Habitat or Replica assets are unavailable in CI, failures in live
integration tests must not be interpreted as pure Python regression.
