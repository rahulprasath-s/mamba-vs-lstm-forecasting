# Data

Place source CSV files in `data/raw/`.

Use `data/processed/` for cleaned, normalized, or feature-engineered CSV files.

Expected minimum CSV shape for `main.py`:

```csv
value
1.0
1.2
1.4
```

Pass a different target column with:

```bash
python main.py --csv data/raw/your_file.csv --target close
```
