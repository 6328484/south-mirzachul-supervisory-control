# Data

## Source

All data describe the South Mirzachul main canal (Uzbek: Janubiy Mirzacho‘l magistral kanali; Russian: Южный Голодностепский магистральный канал) as documented in its passport. The passport was compiled around 2000 and is reproduced in:

Seytov A.J. (2018) PhD dissertation, Appendix 2 (canal passport). See the reference list of the article for the full citation.

No field measurements were used.

## Files

### `structures.csv` — check structures and offtakes (model input)

One row per structure, in chainage order within each kind. The model reads this file (`src/gated_model.py`).

| Column | Meaning |
| --- | --- |
| `kind` | `check` = check structure or head works; `offtake` = offtake or branch canal; `other` = listed in the passport but not modelled |
| `name` | passport name, transliterated (YuR = ЮР, U = У, Kh = Х; "branch" = ветка/tarmoq) |
| `km` | chainage from the head works, km (PK numbers are hectometres: PK145 = 14.5 km) |
| `Q_cap_m3s` | design capacity, m³/s |
| `spans` | number of gate spans |
| `gate_w_m`, `gate_h_m` | width and height of one gate, m |
| `area_ha` | command area, ha |
| `note` | remarks on the transcription |

### `passport_structures_ru.csv` — source transcription

The same passport tables as transcribed in Russian, before processing: `pk_m` is the chainage in metres and `gate` gives "width × height" of the gate in metres. Structures whose two parts are listed separately in the passport (e.g. the right and left channels of PK145) were combined in `structures.csv`.

### Reach data

The geometry and hydraulic data of the ten passport reaches (Table 1 of the article) are given in the `PASSPORT` dictionary in `src/gated_model.py`. These are the length, the discharge at Qmax, the top and bottom widths, the depths at Qmax, Qnorm and Qmin, and the bed slope.

### `canal_route.json` — digitised canal route

`route_latlon` is a list of [latitude, longitude] points (WGS84) along the 123.7-km canal from the head works to the tail structure PK1237; `length_km` is the length of the drawn route. The route was digitised from Google Earth imagery and used to draw Fig. 1. Between km 97 and PK1237 it is interpolated between two located points. The Google Earth screenshots are not redistributed here, so Fig. 1 cannot be regenerated from this repository.

## Processing in the model (Sect. 3.1 of the article)

- The decrease of the passport Qmax along each reach is distributed among the offtakes of that reach in proportion to their design capacities, without exceeding any capacity. The rest (26.1 m³/s in total, in reaches 1 and 3) is treated as distributed seepage and unlisted withdrawals and is excluded from the delivery indicators.
- The offtake demand at Qmax is 259.9 m³/s, and 14.0 m³/s leaves through the tail structure. The five largest branches take 171.4 m³/s (66 %): Right Bayaut 22.8, YuR-10 10.1, Central 102.5, Kurgantepa 13.7 and YuR-18 (Akbulak) 22.4 m³/s.
- The passport gives no sill or bed elevations at the structures, so the bed is continuous. At PK923+02, PK1087+23 and PK1177+29 the effective gate width is increased by factors of 1.16, 1.33 and 1.56. The PK1035 structure is modelled as fully open.
- Level limits at the nine control points: upper = depth at Qmax in the calibrated steady state; lower = passport depth at Qmin = 196 m³/s.
