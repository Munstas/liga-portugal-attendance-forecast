# Liga Portugal Attendance Forecast

Scrapes football matches attendance data from FotMob API for Liga Portugal Betclic (2022-2026)
and predicts attendance for future matches.

## Data Source

**FotMob API** - https://www.fotmob.com

## Known Data Issues

- In match Santa Clara vs FC Porto (2024-08-16), attendance was 7022 and not 100
  as verified on https://www.zerozero.pt/jogo/2024-08-16-santa-clara-fc-porto/10239954

- In matches Sporting CP vs FC Porto (2025-08-30) and Farense vs FC Porto (2025-02-16)
  attendance numbers go over capacity numbers, but it doesn't affect forecast.
  Capacity numbers are just not updated for some stadiums.

- Match Santa Clara vs Moreirense (2025-03-09) has no attendance data.

## How to use

```bash
git clone https://github.com/Munstas/liga-portugal-attendance-forecast.git
pip install -r requirements.txt
python predict.py
```

## Future work

- Get more data for better predictions
- Automatic data updates
- Automatic data updates
