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

## II. Development, Production and Machine Learning Features

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

Note that python tests are incomplete due to resource constraint. Integration tests and end-to-end tests are largely missing. The author essentially opted for manual testing on dev machines instead of coding automated integration and end-to-end tests because of their complexity. Nevertheless, unit tests are relatively comprehensive because they are eaiser to write.

#### II. I. III. Databases
The project does not include any implementation of database instances and their integration with the application logic. In practice, an acyclical flow should be established to preserve and to version control raw, intermediate and final data structures.

Different databases offer different trade-offs. Some of them are listed below:

* Relational Databases
- **PostgreSQL** hosted on AWS is more developer friendly given most developers are familiar with relational databases. The downside is given the tree/graph structure of data structures involved. Relational databases are not just hard to maintain but also miss certain features such as semantic queries.

* Graph Databases
- **Neo4j** specialises storing and querying graph data which are useful tasks given data structures involved. Graph databases suffer from two issues. The first is most developers are not familiar with graph databases and therefore maintainence cost can be high. The second is graph databases are usually more expensive due to their inherent complexity.
- **AWS Neptune** is an alternative in AWS ecosystem.

* Vector Databases
- **Weaviate** offers efficient storage of embedding vectors and semantic search, two powerful features that make vector databases the most appropriate database solution for this project in the author's opinion. Vector databases can be hosted on deployed EKS instances.

#### II. I. IV. Intermediate and Final Data Structures

### II. II. Deployment Features

#### II. II. I. CI/CD
The project uses GitHub Action for on-push and on-pull-request CI workflows. Currently the only active workflow is a python build and test workflow. Constrained by resources, the project does not have a bake workflow and a deployment workflow. A bake workflow can build the application environment to avoid building dependencies being included in inference environment at runtime. A deployment workflow can upload the baked environment to cloud for production.

DevOps requires extensive infrastructure and is largely independent of the application logic itself. Given the project focuses on a machine learning problem and does not demand comprehensive infrastructure that is required to effectively integrate the solution into a production system, MLOps components are simply brushed over with a few bullet points instead of being implemented.

* Automatic Deployment of EKS Instances
- **Terraform** can be used to deploy clusters of ECS instances to host data processing, model training and inference services.

* Vendor Agnostic CI/CD Tools
- **Jenkins** is an alternative to GitHub Action which avoids vendor lock-in.
- **TeamCity** is also an alternative which can be hosted on deployed EKS instances.

* Model Management and Experiment Tracking Tools
* **Weights & Biases** is a tool to manage iterations of machine learning experiments and their results.
* **MLFlow** is a similar tool which can also be hosted on deployed EKS instances.

* Deployment of Micro-services on Deployed EKS Instances
- **Kubernetes** can be used to manage container services hosted by deployed ECS instances at scale.

* Serverless Alternative to Self-managed Clusters
- **AWS Fargate** is a Serverless alternative to eliminate need for maintaining clusters of instances and services run on these clusters. The author prefers Serverless over a massive Infrastructure-as-Code codebase because the author does not poessess such DevOps skills to exploit the benefit of homegrown solutions over serverless cloud computing.

#### II. II. II. Containerisation
Despite the presence of configuration for two containers, only one of which hosts the application itself. The vector database container is currently redundant and is not connected to the application at all.

The docker image for the application, despite its multi-stage setup, requires more optimisation. The image itself contains raw input data which should be hosted either in S3 or a separate database instance. The application currently exists as one single service when multiple services, including database instances and/or S3, should be involved.

Nevertheless, containerised application locks system and python dependencies which improve reproducibility and ease deployment.

### II. III. Machine Learning Features

#### II. III. I. Clustering Algorithms

#### II. III. II. Clustering Result

#### II. III. III. Evaluation Process

## III. Shortcomings and Future Extensions

### III. I. Simple Algorithm

### III. II. Unstable Clustering Result

### III. III. Possibility of Leveraging Existing Knowledge Graph

### III. IV. Possibility of Further Utilising LLM for Feature Engineering and Evaluation

### III. V. Unhosted Data

### III. VI. Incomplete Python and Application Tests

### III. VII. Incomplete Deployment Logic

### III. VIII. Unpublished Python Project

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

8. Call Chat Completion API to provide cluster text labels and to evaluate clustering performance

```sh
```

9. Type LLM's clustering evaluation jsons

```sh
poetry run python -m eos.pipelines.eval_llm_output -psic data/01_raw/llm_sub_industry_clusters.json -pic data/01_raw/llm_industry_clusters.json -psie data/02_intermediate/sub_industry_clusters_eval.json -pie data/02_intermediate/industry_clusters_eval.json
```
