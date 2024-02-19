# Eos

![A version of knowledge graph inferred and assembled by the project](./notebooks/kg.svg)

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
A variety of data structures were used to persist both intermediate (mostly graph elements in the form of dataframes) and final data structures (a serialised networkx graph instance). Most data structures that are neither dataframes nor networkx graphs are python dataclasses serialised and deserialised by a combination of **orjson** and **dacite** libraries. **marshmallow** would be a good candidate for data validation but given the scope of the project, data validation is ignored.

In production, these data structures should be versioned and stored in data services such as S3, DynamoDB, SQL or custom database services such as Weaviate hosted on a EKS cluster.

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

The following shell script serves as the application logic itself if it were to be run in a container.
https://github.com/MacarielAerial/eos/blob/e9969992e812243de225320e6d7ad460694464e1/scripts/project_entry_point.sh

### II. III. Machine Learning Features

#### II. III. I. Clustering Algorithms
The project utilises vanilla K-Means Clustering with limited hyperparameter tuning. The optimal number of clusters is first estimated by a simple heuristics (i.e. the number of clusters is between a half and a quarter of clustering input) and a grid search optimised against silhouette scores.

#### II. III. II. Clustering Result
Clustering result is highly unstable. The number of clusters on each level has about a 15% variation from the average number of clustering after multiple attempts. Clustering result is also not particularly reliable given a small number of examples. Certain clusters have as few as one single member which complicate the evaluation process.

#### II. III. III. Evaluation Process
Given resource constraint, clustering result is evaluated by GPT-4. An attempt was made to invoke Chat Completion API to access the latest GPT-4 model. The attempt failed because the author does not have access to resources required to access GPT-4 model through Chat Completion API.

The author's ChatGPT Plus subscription was used as an alternative. An assistant was created on the author's ChatGPT account to fulfill the role of cluster text labeller and clustering result evaluator. The output of this assistant was copied and pasted over from the web console into the local file system as jsons to fulfill GPT-4's role in the evaluation process.

The following python module details the process to produce input prompt for GPT-4. The process is a combination of explicit instructions to GPT-4 to return json objects of a certain schema and feed of entire clustering result into GPT-4 to solicit text labels and evaluation notes in json format.
https://github.com/MacarielAerial/eos/blob/e9969992e812243de225320e6d7ad460694464e1/src/eos/nodes/query_llm.py

## III. Shortcomings and Future Extensions

### III. I. Simple Algorithm
K-Means clustering is insufficient in its capacity. Given the small scope of example data, K-Means performs reasonably well for such an efficient solution, however, when every inch of performance advantage counts and when input data increases in scale, more complex algorithms are required.

Text embeddingg generation and aggregation processes currently implemented are overly simplistic. A possible improvement is to take one of the following two approaches to produce more dynamic embeddings and to aggregate such embeddings more intelligently.

* Transformer
    - Transformer, essentially a fully connected token-node natural language (mostly) graph with self-attention, can be used to produce embeddings for themes. Themes have extensive descriptions which are well suited as input for transformers. A generalise pre-trained transformer would be a significantly boost already, however, a transformer fine-tuned on industry classification task may produce higher quality and more relevant embeddings. After all, supervised models almost always outperform generalised pre-trained models if the task is well defined.
* Graph Convolutional Neural Network
    - The author is a specialist in graph neural networks. Graph Convolution Neural Network (GCN) is a novel technique which better incorporates topological information in embedding derivation and aggregation. Knowledge graph is a form of data uniquely well suited for GCN and its derivative algorithms. GCN could be used in an unsupervised manner in this project to obtain embeddings without fine tuning a generalised pretrained transformer model whose cost can be prohibitive.

### III. II. Unstable Clustering Result
As mentioned before, the clustering result is highly unstable. There are two possible solutions among many:

* Advanced Clustering Algorithms
    - HDBSCAN is a possible alternative to K-Means Clustering. More advanced clustering algorithms may produce more stable result.
* Better Embeddings
    - Data is as important as optimisation algorithms. As mentioned before, transformer and GCN can possibly obtain higher quality embeddings which indirectly stablise clustering result by further separating entities from each other.

### III. III. Possibility of Leveraging Existing Knowledge Graph
Supplementary information from other related knowledge graph such as [North American Industry Classification System](https://www.census.gov/naics/) or [Industrial Classification for National Economic Activities](https://www.stats.gov.cn/xxgk/tjbz/gjtjbz/201710/t20171017_1758922.html) are both industry classification systems, developed by the world's biggest and the second biggest economies respectively.

Such external knowledge graphs can be embedded alongside project input data. [Past research papers](https://arxiv.org/abs/1905.11605) have shown that high-resource knowledge graphs and benefit low-resource knowledge graphs by projecting their sub-graph structural embeddings over. Although the example paper in the link specifically deals with cross-lingual knowledge graphs, the principle can be used between any knowledge graphs that are similar in their topology.

### III. IV. Possibility of Further Utilising LLM for Feature Engineering and Evaluation
The project's crude use of Chat Completion API can be scrapped altogether and replaced with a more sophisticated procedure which can include memory and state management, chaining and concurrent requests. These features can be implemented through LangChain which would provide more robust embedding derivation and clustering result evaluation services.

LLM can be used as automated clustering result evaluation agent or simply as a solution to produce embeddings. Transformers (other than the one currently used) were mentioned in previous sections to possibly be able to produce higher quality embeddings. For example, GPT-4 Turbo could be fine-tuned into an industry classification expert assistant which could produce embeddings more relevant to the direct task.

### III. V. Unhosted Data
All data structures are stored either on the dev machine or in an unmanaged and static S3 bucket. In production, all data structures should be version controlled along with the stack of Infrastructure-as-Code deployed.

At the very least, the author would have utilised a combination of DynamoDB and S3 to store all data structures used in the project even if they are not the most ideal databases for the task. DynamoDB and S3 have the benefit of being low-cost and easy-to-deploy database solutions.

Additionally, **MLFlow** or **Weights & Biases** can be deployed on managed instances to version control machine learning models (when there is one) given machine learning models are special form of data that should be managed separatedly from a standard data structure.

### III. VI. Incomplete Python and Application Tests
Given time constraint, the test coverage of the project gradually dropped as the deadline creeped closer. Integration and end-to-end tests are completely absent while unit tests only cover a little more than half of the code base.

### III. VII. Incomplete Deployment Logic
Despite being containerised, the project has zero deployment code at the moment. In production, Terraform can be used to deploy EKS instances. Jenkins can be used to build CI/CD pipelines to automate runs. Kubernetes or AWS Fargate can be used to deploy containerised services at scale.

### III. VIII. Unpublished Python Project
The python project can be orchestrated to automatically increment its semantic version numbers and to be automatically deployed onto probably a private PyPi index through services such as AWS CodeArtefact.

### III. IX. Possibility of Automatic Documentation
Sphinx can possibly be used to automatically produce API documentation. Although the code base is minimally documented, the project's convenient structure makes it well suited to be documented by frameworks such as Sphinx which can reduce maintainence overhead.

### III. X. Insufficient Error Handling
The project uses principles of error prevention and "fail-fast". Minimal error handling code such as try-except blocks is implemented, however, the relatively comprehensive tests along with modular and well structured code ideally would fail the run at the point of errors.

Functions in the project, combined with type hints, take liberty in assuming schema of input data structures. The idea is if the input structure is malformed, the process would exist immediately with errors instead of sliently failing.

## IV. Code Examples

1. Create a typed representation of raw data

Raw data can be converted into python dataclasses to make sure a change upstream would immediately crash the process on the data access level. This is to prevent silent failure that cascades downstream.

```sh
poetry run python -m eos.pipelines.type_raw_source_themes -prst data/01_raw/industrial_business_theme_descriptions.jsonl -pst data/02_intermediate/source_themes.json
```

2. Parses graph elements from typed data

Raw data contains entity and relation information in one single object. Separation of entity and relation information allows both to be modified with minimal effect on each other.

```sh
poetry run python -m eos.pipelines.source_themes_to_element_dfs -pst data/02_intermediate/source_themes.json -pnd data/03_primary/node_dfs.json -ped data/03_primary/edge_dfs.json
```

3. Encode graph element features

Raw features are text which is unstructured and cannot be used directly as features. Vectorisation is done with a standard text2vec model.

```sh
poetry run python -m eos.pipelines.encode_features -pnd data/03_primary/node_dfs.json -pst data/01_raw/all-MiniLM-L6-v2/ -pdfe data/04_feature/
```

4. (Optional) Store encodings in a vector database

The concept was to enable efficient vector storage, retrieval and semantic queries. Given time constraint, this approach was abandoned in favour of file-based data storage.

```sh
```

5. Cluster themes into sub industries and cluster sub industries into industries

A simple clustering algorithm is used in favour of more complex ones to establish a baseline and to speed up development.

```sh
poetry run python -m eos.pipelines.cluster_for_sub_and_industries -pte data/04_feature/theme.npy -pde data/04_feature/description.npy -psil data/04_feature/sub_industry_label.npy -pil data/04_feature/industry_label.npy
```

6. Parse intermediate layer graph elements from base layer elements and clustering result

Clustering result is integrated back into the knowledge graph as additionaly entities and relations.

```sh
poetry run python -m eos.pipelines.parse_interm_layer_elements -pbnd data/03_primary/node_dfs.json -pbed data/03_primary/edge_dfs.json -psil data/04_feature/sub_industry_label.npy -pil data/04_feature/industry_label.npy -pind data/04_feature/interm_node_dfs.json -pied data/04_feature/interm_edge_dfs.json
```

7. Construct a knowledge graph from graph elements of all types

First pass to build a knowledge graph which neither has text labels associated with sub industry and industry level nodes nor is evaluated in any way.

```sh
poetry run python -m eos.pipelines.assemble_kg -pnd data/04_feature/interm_node_dfs.json -ped data/04_feature/interm_edge_dfs.json -png data/04_feature/nx_g.json
```

8. Call Chat Completion API to provide cluster text labels and to evaluate clustering performance

OpenAI's Chat Completion API or ChatGPT's GPT-4 assistant is invoked to produce text labels for clustering result and one to three sentence summary on GPT-4's comment on clustering performance.

```sh
# Insufficient spending of the author's own account prevents the author from accessing GPT-4 through Chat Completion API
# This stage is replaced with manual requests through ChatGPT web console
```

9. Type LLM's clustering evaluation jsons

LLM's raw output is typed given Chat Completion API does not always return a json object with valid schema.

```sh
poetry run python -m eos.pipelines.eval_llm_output -psic data/01_raw/llm_sub_industry_clusters.json -pic data/01_raw/llm_industry_clusters.json -psie data/02_intermediate/sub_industry_clusters_eval.json -pie data/02_intermediate/industry_clusters_eval.json
```

10. Augment cluster node dataframes with typed LLM output

```sh
poetry run python -m eos.pipelines.augment_element_dfs_with_llm -pbnd data/04_feature/interm_node_dfs.json -psie data/02_intermediate/sub_industry_clusters_eval.json -pie data/02_intermediate/industry_clusters_eval.json -plnd data/04_feature/llm_node_dfs.json
```

11. Construct a second knowledge graph from elements supplemented with LLM output

```sh
poetry run python -m eos.pipelines.assemble_kg -pnd data/04_feature/llm_node_dfs.json -ped data/04_feature/interm_edge_dfs.json -png data/04_feature/llm_nx_g.json
```
