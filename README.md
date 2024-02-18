# Eos

## I. Executive Summary
Eos is an application which given the root and the terminal layer of an industry classification taxonomy, infers the two intermediate layers of taxonomy.

The problem calls for clustering of "themes" which are categories of companies in the bottom of the taxonomy to produce "sub-industries" which are categories of "themes". Then "industries" are produced as the result of clustering "sub-industries". Ultimately, "industries" are linked to "sectors" which are on the top of the taxonomy. In the scope of this project, all "themes" supplied belong to a single sector. The problem would be more complicated if neither "sectors" nor links between "themes" and "sectors" are given.

The solution is simple two-stage K-Means clustering of text embeddings of "themes". "Themes" are clustered into "sub-industries" in the first stage and "sub-industries" into "industries" in the second stage. Given the example data belongs to one single sector, "industries" are automatically linked to the only "sector" available.

Despite the algorithm's simplicity, the main purpose of the project is to create an application ready to be integrated into a production system of a modern software product. Given the proper structure, the algorithm within can be easily swapped while keeping deployment, serving and monitoring logic mostly unchanged.

Four sections of documentation followed addressed each of these issues in turn:
* Section II outlines project features that cater towards other developers whose responsibilities are either to further develop the application logic or to deploy the application as a part of a production system.
* Section III outlines major issues within the current version and possible remedies that may be implemented in the future.
* Section IV contains code examples required to execute individual modules of the application. This section is aimmed at developers who wish to further extend existing application logic.

## II. Development and Production Features

### II. I. Development Features

#### II. I. I. Python Application Structure
The python application logic is structured based on a popular open source data science framework **Kedro** developed by **QuantumBlack**. The python code structure also bears significant resemblance to other data science framework such as **Dagster**, however, the project chose to use these frameworks as inspiration instead of copying them altogether given unnecessary dependency lock-in compromises project maintainence for other developers who do not whole-heartedly subscribe to a particular framework.

Specifically, the project's python code is divided into three sections:
1. Nodes: Functions that perform a specific task, usually transforming one set of data structures into another.
2. Data Interfaces: Classes that handle data access logic. Their scope in this project is limited to in-memory objects, files and local database instances, however, they could be extended to interact with APIs.
3. Pipelines: Functions which usually invoke a task processing node (i.e. "nodes") and two data access nodes (i.e. "data interfaces"). Pipelines are essentially chains of data access logic and transformation logic which serve as a mid-level interface for developers. Pipelines also come with their own CLI which can be access with "python -m abc.efg -arg arg_val ..." pattern.

#### II. I. II. Code Quality Control
The project uses a variety of lint and test tools to maintain code quality. Here's a list of tools used in the project along with their purpose.

* Lint Tools
- black (automatic python code reformatting)
- isort (sort import orders)
- flake8 (pep8 standard compliance check)
- mypy (python type hints)
- ruff (a mixture of all of the above)
- semgrep (static code analysis for security vulnerability scan)
- shellcheck (shell script quality check)
- yamllint (yaml syntax check)

* Test Tools
- pytest (python code tests)
- coverage (python code coverage check)

#### II. I. III. CI/CD
The project uses GitHub Action for on-push and on-pull-request CI workflows. Currently the only active workflow is a python build and test workflow. Constrained by resources, the project does not have a bake workflow and a deployment workflow. A bake workflow can build the application environment to avoid building dependencies being included in inference environment at runtime. A deployment workflow can upload the baked environment to cloud for production.

DevOps requires extensive infrastructure and is largely independent of the application logic itself. Given the project focuses on a machine learning problem and does not demand comprehensive infrastructure that is required to effectively integrate the solution into a production system, MLOps components are simply brushed over with a few bullet points instead of being implemented.

* Automatic Deployment of EKS Instances
- **Terraform** can be used to deploy clusters of ECS instances to host data processing, model training and inference services.

* Deployment of Micro-services on Deployed EKS instances
- **Kubernetes** can be used to manage container services hosted by deployed ECS instances at scale.

* Deployment of Micro-services with Serverless Framework
- **AWS Fargate** is a Serverless alternative to eliminate need for maintaining clusters of instances and services run on these clusters. The author prefers Serverless over a massive Infrastructure-as-Code codebase because the author prefers to work 

## III. Shortcomings and Future Extensions

## IV. Code Examples

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
