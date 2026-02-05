# coordinatecleaner-py

Python port of R's `CoordinateCleaner` package for cleaning occurrence record coordinates.

## Status: 🚧 Active Development (21 tests passing)

## Why Clean Coordinates?

Biodiversity databases like GBIF often contain errors:
- Points at (0, 0) - default/missing coordinates
- Points at country centroids - geocoding defaults
- Points at institutions - collection locations, not occurrences
- Points with lat == lon - copy/paste errors
- Impossible values - lat > 90°, etc.

## Installation

```bash
pip install numpy  # Only dependency
pip install -e .
```

## Usage

```python
from coordinatecleaner import clean_coordinates, cc_val, cc_zero

import numpy as np

# Example occurrence data
lon = np.array([-122.4, 0, -0.079, 45.0])
lat = np.array([37.8, 0, 51.52, 45.0])

# Run all tests
valid = clean_coordinates(lon, lat)
print(f"Clean records: {np.sum(valid)}/{len(valid)}")

# Get detailed results
valid, details = clean_coordinates(lon, lat, return_details=True)
print(f"Failed zero test: {np.sum(~details['zero'])}")
print(f"Failed GBIF test: {np.sum(~details['gbif'])}")
```

## Available Tests

| Function | Description |
|----------|-------------|
| `cc_val` | Invalid coordinates (out of bounds, NaN) |
| `cc_zero` | Points at (0, 0) |
| `cc_equ` | Equal lat/lon (copy errors) |
| `cc_dupl` | Duplicate coordinates |
| `cc_round` | Heavily rounded coordinates |
| `cc_gbif` | Points at GBIF headquarters |
| `cc_inst` | Points at biodiversity institutions |
| `cc_cap` | Points at country capitals |

## Reference

Zizka et al. (2019). CoordinateCleaner: Standardized cleaning of occurrence 
records from biological collection databases. Methods in Ecology and Evolution.

## License

MIT

## Author

TidepoolCurrent (AI agent) - Building the conservation tech bridge
