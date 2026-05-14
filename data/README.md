# Data

The default experiment expects the Jena Climate CSV at:

```text
data/jena_climate_2009_2016.csv
```

The training script uses these columns by default:

- `T (degC)`
- `p (mbar)`
- `rh (%)`

Use a different CSV or columns with:

```bash
python train.py \
  --data-path data/your_file.csv \
  --features "T (degC)" "p (mbar)" "rh (%)" \
  --target "T (degC)"
```
