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

4. (Optional) Store encodings in a vector database

```sh
```

5. Cluster themes into sub industries and cluster sub industries into industries

```sh
poetry run python -m eos.pipelines.cluster_for_sub_and_industries -pte data/04_feature/theme.npy -pde data/04_feature/description.npy -psil data/04_feature/sub_industry_label.npy -pil data/04_feature/industry_label.npy
```

6. Parse intermediate layer graph elements from base layer elements and clustering result

```sh
poetry run python -m eos.pipelines.parse_interm_layer_elements -pbnd data/03_primary/node_dfs.json -pbed data/03_primary/edge_dfs.json -psil data/04_feature/sub_industry_label.npy -pil data/04_feature/industry_label.npy -pind data/04_feature/interm_node_dfs.json -pied data/04_feature/interm_edge_dfs.json
```

7. Construct a knowledge graph from graph elements of all types

```sh
poetry run python -m eos.pipelines.assemble_kg -pnd data/04_feature/interm_node_dfs.json -ped data/04_feature/interm_edge_dfs.json -png data/04_feature/nx_g.json
```
