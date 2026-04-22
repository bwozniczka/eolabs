# Earth Engine function inventory for `SAR_2_restructured_solution.ipynb`
This handout is intentionally compact. The “options” column lists the most common arguments or variants students are likely to need, not the full API.

## 1) Build/filter a collection:
| Function | What it does here | Common options / arguments |
|---|---|---|
| `ee.ImageCollection(asset_id)` | Load an image collection asset. | asset ID string, existing image/list/computed object |
| `ImageCollection.filterBounds(region)` | Keep images intersecting the AOI. | geometry / feature / feature collection |
| `ImageCollection.filterDate(start, end)` | Keep images in a time window. | start/end as strings, ee.Date, or milliseconds |
| `ImageCollection.filter(filter_obj)` | Apply a custom metadata/date/spatial filter. | any ee.Filter |
| `ee.Filter.eq(name, value)` | Filter metadata equal to a value. | property name + comparison value |
| `ee.Filter.listContains(name, value)` | Require that a list-valued property contains a value. | left field/list + right value |
| `ImageCollection.size()` | Count images in the collection. | no args |
| `ee.Filter.calendarRange(...)` | Filter by month, DOY, weekday, etc., instead of explicit start/end dates. | Seasonal subsets |
| `ee.Filter.inList(...)` | Match metadata against a list of allowed values. | Several acceptable orbit/platform values |
| `filterBounds(...) ↔ ee.Filter.bounds(...)` | Same spatial idea; use ee.Filter.bounds when you want the filter object explicitly. | More modular filter chains |
| `filterDate(...) ↔ ee.Filter.date(...)` | Same date filtering but explicit as a filter object. | Reusable filter objects |

## 2) Make composites:
| Function | What it does here | Common options / arguments |
|---|---|---|
| `ImageCollection.median()` | Create a per-pixel median composite. | no args; bands matched by name |
| `ImageCollection.mean()` | Per-pixel mean composite instead of median. | When averaging is acceptable |
| `ImageCollection.min()/max()` | Per-pixel min/max composite. | Envelope or extreme-value views |
| `ImageCollection.mode()` | Per-pixel most common value. | Categorical bands/classes |
| `ImageCollection.sort(property, ascending)` | Order a collection before taking first()/mosaic(). | Best/worst scene selection |
| `ImageCollection.first()` | Take the first image after sorting/filtering. | Single-scene examples |
| `ImageCollection.mosaic()` | Stack images by order and mask, rather than reducing statistically. | Visual gap filling |
| `ImageCollection.qualityMosaic(score_band)` | Pick the best pixel based on a quality band. | Cloud/quality-driven composites |
## 3) Band math:
| Function | What it does here | Common options / arguments |
|---|---|---|
| `Image.select(bands)` | Choose one or more bands. | names, indices, regex, optional rename list |
| `Image.subtract(other)` | Subtract one image/band from another. | other image or scalar |
| `Image.rename(new_name)` | Rename output band(s). | one string, varargs, or list of names |
| `Image.normalizedDifference([b1, b2])` | Compute (b1-b2)/(b1+b2) directly. | NDVI/NDWI-style indices |
| `Image.expression(expr, map)` | Write band math as an expression instead of chained arithmetic. | Complex formulas |
| `Image.addBands(other, overwrite=...)` | Append derived bands instead of keeping them separate. | Multi-band workflow building 
## 4) Masks and thresholds:
| Function | What it does here | Common options / arguments |
|---|---|---|
| `Image.where(test, value)` | Conditional replacement of pixels. | Threshold-based recoding |
| `Image.gte()/lte()/neq()` | Other comparison operators parallel to gt()/lt(). | Inclusive or not-equal tests |
| `Image.Or()/Not()` | Other logical operators parallel to And(). | More complex masks |
| `Image.selfMask()` | Mask zeros using the image’s own values. | Cleaner display of boolean masks |
| `Image.unmask(value, sameFootprint=...)` | Fill previously masked pixels. | Gap filling / export preparation |
| `Image.lt(value)` | Pixelwise less-than comparison; returns boolean image. | image or scalar |
| `Image.gt(value)` | Pixelwise greater-than comparison; returns boolean image. | image or scalar |
| `Image.And(other)` | Logical AND of boolean images. | other image or scalar |
| `Image.updateMask(mask)` | Keep only pixels where the mask is nonzero. | mask image; 0/1 or floating opacity |

## 5) Summaries/statistics:
| Function | What it does here | Common options / arguments |
|---|---|---|
| `Image.reduceRegion(...)` | Summarize pixel values over one region. | reducer, geometry, scale, crs/crsTransform, maxPixels, bestEffort, tileScale |
| `ee.Reducer.fixedHistogram(min, max, steps)` | Build a fixed-bin histogram reducer. | min, max, number of bins; optional cumulative on full doc |
| `ee.Reducer.mean()` | Mean reducer. | no args |
| `ee.Reducer.median()` | Median reducer. | no args |
| `ee.Reducer.stdDev()` | Standard deviation reducer. | no args |
| `ee.Reducer.max()` | Maximum-value reducer. | optional numInputs |
| `Reducer.combine(other, sharedInputs=True)` | Run multiple reducers in one reduceRegion call. | reducer2, outputPrefix, sharedInputs |
| `Reducer.percentile([...])` | Return percentiles instead of mean/median/max. | Robust spread summaries |
| `Reducer.group(...)` | Group reducer output by class/category. | Class-wise summaries |
| `Image.reduceRegions(...)` | Summarize over many polygons at once instead of one AOI. | Per-feature zonal stats |
| `Image.reduceNeighborhood(...)` | Neighborhood/window reducer instead of whole-region reducer. | Texture/local averaging |
| `Image.sample(...)` | Extract sampled pixels as features. | Training tables / quick inspection |
| `Image.sampleRegions(...)` | Sample pixels at vector features/polygons. | Training and validation datasets |

## 6) Server to client:
| Function | What it does here | Common options / arguments |
|---|---|---|
| `ImageCollection.aggregate_array(property)` | Pull one property from all images into a server-side list. | property name |
| `ee.List(obj)` | Cast/construct a server-side list. | Python list or computed object |
| `Dictionary.get(key)` | Extract one value from a server-side dictionary, e.g. reduceRegion output. | key, optional defaultValue 
| `ComputedObject.getInfo()` | Bring a server-side object to the client immediately. | optional callback in some contexts; synchronous in Python |
| `Dictionary.getNumber()/getString()/getArray()` | Typed alternatives to generic get(). | Cleaner downstream typing |

## 7) Other useful functions:
| Function | What it does here | Common options / arguments |
|---|---|---|
| `ee.Initialize(project=...)` | Authenticate/initialize the Python client for a specific EE project. | project=..., opt_url/baseurl, credentials handled outside or implicitly |
| `ee.Geometry.Rectangle(coords)` | Build a rectangular AOI geometry. | 4 coords or [xmin, ymin, xmax, ymax]; optional proj, geodesic, evenOdd |
| `Image.clip(region)` | Mask pixels outside the AOI. | geometry or feature |
| `Image.set({...})` | Attach metadata/properties to an image. | dictionary or key/value pairs |
