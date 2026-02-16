# Liga Portugal Attendance Scraper

Scrapes attendance and stadium capacity from FotMob API for Liga Portugal Betclic
(2022-2026). 1200+ matches processed.

## 🔗 Data Source

**FotMob API** - https://www.fotmob.com

## Known Data Issues

- In match Santa Clara vs FC Porto (2024-08-16), attendance was 7022 and not 100
  as verified on https://www.zerozero.pt/jogo/2024-08-16-santa-clara-fc-porto/10239954

- In matches Sporting CP vs FC Porto (2025-08-30) and Farense vs FC Porto (2025-02-16)
  attendance numbers go over capacity numbers, but it doesn't affect forecast.
  Capacity numbers are just not uptaded for some stadiums.

## How to use

```bash
pip install -r requirements.txt
python scraper.py
```

## To do

Train model to predict attendace in future matches
