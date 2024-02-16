# Eos

## Executive Summary
Eos is an application which given the root and the terminal layer of an industry classification taxonomy, infers the two intermediate layers of taxonomy.

## Code Examples

1. Create a typed representation of raw data

```sh
poetry run python -m eos.pipelines.type_raw_source_themes -prst data/01_raw/industrial_business_theme_descriptions.jsonl -pst data/02_intermediate/source_themes.json
```

2. Parses graph elements from typed data

```sh
poetry run python -m eos.pipelines.source_themes_to_element_dfs -pst data/02_intermediate/source_themes.json -pnd data/03_primary/node_dfs.json -ped data/03_primary/edge_dfs.json
```

3. Encode graph element features

```sh
poetry run python -m eos.pipelines.encode_features -pnd data/03_primary/node_dfs.json -pst data/01_raw/all-MiniLM-L6-v2/ -pdfe data/04_feature/
```
