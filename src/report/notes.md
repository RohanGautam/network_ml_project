# Project proposal

I'm working on a project for a course with the following document as a starting point.



## NBA Players Trajectories

EE-452: Network Machine Learning

EPFL - Spring 2025/2026

### 1 Introduction

Human motion forecasting aims to predict the future trajectories and poses of humans given observations of their past movements. Sports players' trajectories are a good example of such a setting. In this project, we will focus on basketball players from the National Basketball Association (NBA). The player and ball trajectories exhibit complex patterns of coordinated movement, rapid changes in direction, and strong interaction dynamics between multiple agents, providing a high-dimensional benchmark for studying multi-agent motion prediction and learning interaction-aware representations.

However, forecasting human motion in multi-agent environments presents several key challenges. First, human motion is inherently stochastic: multiple plausible futures may exist for the same observed history. Furthermore, interactions between agents introduce strong dependencies, where the motion of one individual influences and is influenced by the behavior of others. Finally, modeling data with high dimensionality and temporal structures requires models that can capture both spatial relationships and long-range temporal dependencies.

Graph-based deep learning models are a powerful framework for representing structured interactions. By modeling entities as nodes and relationships as edges, graph neural networks (GNNs) enable the propagation of information across the graph, allowing the model to capture dependencies between agents while jointly modeling temporal evolution. Graph-based methods thus provide a flexible way to learn interaction-aware representations directly from data, making them particularly well-suited for multi-agent motion prediction scenarios.

**Main Objective:** This project aims to explore the utility of graph-based methods for human motion forecasting, and more precisely NBA players' trajectories. Students should compare graph-based and non-graph-based approaches, evaluating their advantages and limitations. Through this comparison, students will gain insights into how graph structures influence model performance and interpretability. These insights should be clearly articulated in the project report and presentation.

### 2 Dataset & Task Definition

In this project, we use an alternate version of the NBA highlight dataset from the SocialVAE paper [4]. It consists of more than 6K highlight plays from the 2015-2016 season. The player and ball movements have been recorded with a fixed frame rate for a variable number of frames, ranging from 20 to more than 100 steps. An example of such a trajectory is shown in Figure 1. The data is available on the portal of the Kaggle competition (see section 3).

**Dataset:** The data is stored as PyTorch tensors (.pt) with dimension  $L \times N \times D$ , where  $L$  is the sequence length,  $N$  the number of entities and  $D$  the feature dimension. While  $N$  and  $D$  are fixed for the whole dataset, the length of sequences  $L$  may vary among samples. The features are the concatenation of the spatial coordinates, a binary variable *isplayer* indicating whether the entity is a player or the ball, and finally a variable *team* encoding the team membership ( $-1 = \text{Team A}$ ,  $0 = \text{Ball}$ ,  $1 = \text{Team B}$ ). The final feature vector is thus:  $[x, y, \text{isplayer}, \text{team}]$ . Although *isplayer* and *team* are redundant, both are provided to facilitate filtering. Since these fields are not used for evaluation, they can be dropped during pre-processing if convenient. Note that only the spatial coordinates are dynamic features, whereas the others are static over time.

![Figure 1: Illustration of the trajectory of players and the ball during one highlight. The plot shows a soccer field with x and y axes. A central circle represents the ball, and several dashed lines represent player trajectories. A legend in the top right corner indicates 'team' with values -1 (red), 0 (green), and 1 (blue).](81bf7a5187d196cc72844283e3aeec49_img.jpg)

Figure 1: Illustration of the trajectory of players and the ball during one highlight. The plot shows a soccer field with x and y axes. A central circle represents the ball, and several dashed lines represent player trajectories. A legend in the top right corner indicates 'team' with values -1 (red), 0 (green), and 1 (blue).

Figure 1: Illustration of the trajectory of players and the ball during one highlight. The players are associated with team  $-1$  or  $1$ , the ball with group  $0$ .

**Objective:** The goal of the project is, given a context sequence of length  $C$ , to predict the trajectory of each entity (players and ball) for a horizon of length  $H$ . In our setup, we define  $C = 8$  and  $H = 12$ , which gives an overall window of 20 time steps. Note that the window size is smaller than most sequences. You can play with it to sample different windows inside each sequence. The evaluation metric that we will use is the ADE (average discrepancy error), defined as the average Mean Square Error (MSE) over the predicted time steps in the horizon  $H$ :

$$\text{ADE} = \frac{1}{H \times N} \sum_{t=C+1}^{C+H} \sum_n \text{MSE}(\hat{z}_t^n, z_t^n), \quad (1)$$

where  $\hat{z}_t^n = (\hat{x}_t^n, \hat{y}_t^n) \in \mathbb{R}^2$  corresponds to the predicted position of the entity  $n$  (player or ball) at timestep  $t$ , and  $z_t^n = (x_t^n, y_t^n) \in \mathbb{R}^2$  to the corresponding ground truth. Therefore, the evaluation metric is computed only on the spatial coordinates  $(x, y)$ , not on static features.

**Intended Approach:** For this task, graph-based representations can be particularly useful to capture spatial properties, interactions between players, and the heterogeneity of entities on the field. The students are encouraged to explore different graph constructions, drawing inspiration from both the geometry and semantics of the problem. Importantly, we expect all proposed solutions to include a graph-based component as a core element of the methodology. Students may optionally compare their approach against alternative baselines, such as vision-based methods, but submissions that do not incorporate a graph-based solution will not be considered aligned with the objectives of the project.

### 3 Kaggle competition

To enhance engagement and encourage experimentation, we will host a [Kaggle competition](#) where students can test their baselines in a competitive setting. This competition provides an opportunity to benchmark models and refine approaches. Additionally, to ensure a meaningful challenge, we will award a bonus to the top-performing groups. See Section 7 for more details.

While the Kaggle leaderboard provides a way to compare performance across teams, it should primarily serve as a reference rather than the sole evaluation metric. We strongly encourage groups to conduct and report local cross-validation performances using a validation set within the provided training data. This ensures that model selection is not solely driven by Kaggle rankings but is guided by robust internal evaluation practices.

### 4 Provided Pipeline

We provide a python notebook associated with this PDF to help you start with the project. It contains visualization tools and a basic training pipeline to train a naive temporal baseline, without graph prior, whose score is highlighted as *solution.csv* on the Kaggle leaderboard. You are invited to

take inspiration from this pipeline or to design your own! Many improvements can be made, e.g., on the sampling process, model architecture or training procedure, among others. The NBA dataset has been used in many different studies ([4, 3, 1, 2]), do not hesitate to read them to gather some ideas on how to design an interesting architecture and transfer their modeling to graph methods!

### 5 Computational Resources

All enrolled students will have access to GPU resources on the SCITAS cluster, Izar. If you are not familiar with SCITAS, please review the documentation available at [SCITAS Documentation](#). The dataset (approximately 1 GB) will be available on the cluster. Alternatively, you may choose to use your own computational resources or Google Colab.

### 6 Project Expectations & Deliverables

Students are expected to submit **three deliverables: a report, a presentation, and the corresponding code.**

### 6.1 Code

We recommend using libraries already utilized in the exercise session notebooks (such as *PyTorch*, *PyTorch Geometric*, *NetworkX*, ...), but students are free to use any existing open-source code to build their models. The code should be well-documented and reproducible.

In addition to the code submission, students must provide a short screen-recorded video with voice narration presenting their implementation (**maximum duration: 5 minutes**). The recording should briefly explain the structure and organization of the codebase, highlight clarity and documentation, and include a short demonstration showing the code running. Students may use any screen recording software of their choice. The video must be submitted in MP4 format.

### 6.2 Report

The report should follow the official [ICLR 2026](#) format. Each group must submit a strictly 4-page report, excluding references, following a structured format similar to a scientific paper. The report should include the following sections:

- **Introduction:** Problem motivation and objectives.
- **Method:** Approach and technical details.
- **Results:** Experiments, findings, and discussion.
- **Conclusion:** Summary and insights.

### 6.3 Presentation

Each group will give a presentation of up to 10 minutes, summarizing their work. The structure of the presentation should closely follow the report, covering the problem statement, methodology, key findings, and conclusions. A 5-minute Q&A will follow the presentation.

### 6.4 Submission Deadlines

- **Report and Code Submission:** Due on Moodle by June 10th, 2026.
- **Late Submission Policy:** Submissions receive full marks if submitted before the deadline, 90% if within 24 hours after the deadline, 80% if within 48 hours, and 0% thereafter.
- **Presentation Dates:** June 15th–16th, 2026.

## 7 Evaluation Criteria

The project grade will be based on three components: presentation, code, and report. The grading breakdown is shown in Table 1. All team members will receive the same grade.

### 7.1 Bonus Points

We will award bonus points to the top 10 performing teams in the Kaggle competition. The bonus will be distributed in a linearly spaced manner, with the 1st team receiving the maximum bonus of 0.5 and the 10th team receiving the minimum of 0.05, with intermediate teams awarded proportionally (linearly) decreasing bonuses. These bonus points will be added to the final project grade (capped at 6).

| Category                             | Criterion (%)                                                 | Poor (1)             | Limited (2)                      | Satisfactory (3)   | Good (4)                     | Excellent (5)                     |
|--------------------------------------|---------------------------------------------------------------|----------------------|----------------------------------|--------------------|------------------------------|-----------------------------------|
| <b>Presentation and Report (80%)</b> | Description of methods (20%)                                  | Incorrect or missing | Incomplete explanation           | Basic description  | Mostly clear with minor gaps | Clear, precise, technically sound |
|                                      | Originality of the approach (10%)                             | No baselines         | Limited novelty (just baselines) | Standard approach  | Some originality             | Highly original                   |
|                                      | Interpretation of results (25%)                               | Misinterpretation    | Superficial comments             | Basic discussion   | Good interpretation          | Deep and critical analysis        |
|                                      | <b>Presentation:</b> communication and clarity (15%)          | Unclear              | Hard to follow                   | Understandable     | Clear overall                | Very clear and engaging           |
| <b>Code (20%)</b>                    | <b>Report:</b> writing quality and overall presentation (10%) | Poorly written       | Many issues                      | Acceptable writing | Well written                 | Excellent structure and writing   |
|                                      | Functionality (10%)                                           | Does not run         | Major problems                   | Runs with fixes    | Minor issues                 | Runs without issues               |
|                                      | Code quality and documentation (10%)                          | Poor quality         | Hard to read                     | Adequate           | Mostly clean                 | Very clean and well documented    |

Table 1: Grading breakdown.

## References

- [1] Fu, Y., Yan, Q., Wang, L., Li, K., Liao, R.: Moflow: One-step flow matching for human trajectory forecasting via implicit maximum likelihood estimation based distillation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 17282–17293 (2025), <https://arxiv.org/abs/2503.09950>
- [2] Gao, Y., Luan, P.C., Messaoud, K., Feng, L., Alahi, A.: Omnitraj: Pre-training on heterogeneous data for adaptive and zero-shot human trajectory prediction. arXiv preprint arXiv:2507.23657 (2025), <https://arxiv.org/abs/2507.23657>
- [3] Sestak, F., Toshev, A., Fürst, A., Klambauer, G., Mayr, A., Brandstetter, J.: Lam-slide: Latent space modeling of spatial dynamical systems via linked entities. arXiv preprint arXiv:2502.12128 (2025), <https://arxiv.org/abs/2502.12128>
- [4] Xu, P., Hayet, J.B., Karamouzas, I.: Socialvae: Human trajectory prediction using timewise latents. In: European Conference on Computer Vision. pp. 511–528. Springer (2022), <https://arxiv.org/abs/2203.08207>



---


# what we did and discussed

Links
https://www.kaggle.com/t/bdb1639f766847d4b254975917ecc92e the competition
https://github.com/RohanGautam/network_ml_project 
https://github.com/THUMNLab/AutoGL maybe
https://github.com/traja-team/traja useful library for trajectory manipulation, feature extraction
https://github.com/astral-sh/uv python environment management
https://github.com/Chenwangxing/Review-of-PTP-Based-on-GNNs?tab=readme-ov-file#42-Multi-type-agent-heterogeneous-graph-models





Literature review
Lam-slide: Latent space modeling of spatial dynamical systems via linked entities (rohan) 
Maps spatial locations of players to latent space to get a “game state”, trajectory evolved in the latent space, a flow-based network to decode latent states to trajectory information.
Not graph based directly, uses cross attention (to map to latent space as well as during the decoding process). But again, transformers can be seen as GNNs (https://arxiv.org/abs/2506.22084), with cross attention simulating message passing on a fully connected graph. Not sure if this can directly be justified for the project tho.
Two stage training process, one for learning spatial representations and another for learning temporal dynamics.
Has baselines on the NBA dataset, but while the model is trained with the whole data, evaluation is done on scoring/rebounding splits (similar to socialVAE i think)
	
Splits from https://github.com/xupei0610/SocialVAE 
Uses minADE/minFDE metrics
Code @ https://github.com/ml-jku/LaM-SLidE 
8 frame input, 12 frame prediction window as context -> same as project requirements
Gao, Y., Luan, P.C., Messaoud, K., Feng, L., Alahi, A.: Omnitraj: Pre-training on heterogeneous data for adaptive and zero-shot human trajectory prediction (Angana)
Transformer based, no graph component
Main strength - zero-shot, generalises to new settings 
Generates specific embeddings for the FPS, uses a cross-modal transformer internally that combines modality-specific information and multimodal info like 3d pose estimation, bounding boxes, etc. Trained on diverse datasets (different setups, FPS, etc). 
Code available - can use it as a separate experiment, but the result is likely to be very good
SocialVAE: Human Trajectory Prediction using Timewise Latents (Thibaut)
Exploitation of Hidden Context in Dynamic  Movement Forecasting: A Neural Network Journey  from Recurrent to Graph Neural Networks and  General Purpose Transformers : Schelenz et al (rohan)
This is a review paper specifically for NBA player trajectory forecasting - convenient
Hybrid LSTM outperforms transformers, GNNs
Has sufficient detail about dataset preprocessing
Apart from ADE+FDE, uses AAE and FAE (angular errors)


0.48s is 12 frames, more directly comparable to lam-slide, but data seems different so not really
GNN advantages: interpretability
CNN based temporal embeddings before passing to transformers/GNNs
To be fair, they only tried vanilla GNNs - for example, LaM-slide is a more sophisticated model and it’s metrics (linked before) are better than the metrics provided here (for scoring at least) - they didn't do rebounding/scoring split. Cant compare because minADE is not same as ADE, and there is a slight variation in dataset (todo: investigate more)

Graph-based papers:
Social-stgcnn: A social spatio-temporal graph convolutional neural network for human trajectory prediction (referenced in Omnitraj) 
Choose graph representation 
Groupnet: Multiscale hypergraph neural networks for trajectory prediction with relational reasoning. (referenced in Omnitraj) - hypergraph approach 
GroupNet -> MART, STGFormer (transformer based)
MART: MultiscAle Relational Transformer Networks for Multi-agent Trajectory Prediction: Improvement over Groupnet by combining transformers
Hyper-STTN: Hypergraph Augmented Spatial-Temporal Transformer Network for Trajectory Prediction. No publicly available code, but maybe can email them
EvolveGraph: Multi-Agent Trajectory Prediction with Dynamic Relational Reasoning. Widely cited, but no publicly available code
Graph-based diffusion model (?)

Neural Relational Inference for Interacting Systems 
VAE with GNNs for encoding/decoding
Good for interpretability 

Potential flow:
Setup (rohan)
Metrics, data exploration, splits
Non graph baseline (rohan)
GRU
Transformer
Fixed graph structure (thibaut)
SGAT, STGAT, Social STGCNN
Hypergraph groups (angana)
Groupnet 
For best result on NBA, combined with Collaborative Motion Prediction via Neural Motion Message Passing. Potential idea: combine groupnet’s encoder with other approaches
Code available
Improvement on Groupnet: MART (uses transformer)
Learnt graph structure
Neural relational inference https://arxiv.org/pdf/1802.04687 
Dynamic graph, changes with time (nahush)
EvolveGraph 

https://www.sciencedirect.com/science/article/pii/S0952197625001253?via%3Dihub
Equivariance (rohan)
EqMotion(2023)
Can add this to best performing models?



MSTT: A Multi-Spatio-Temporal Graph Attention Model for Pedestrian Trajectory Prediction 

SocialCircle+: Learning the Angle-based Conditioned Interaction Representation for Pedestrian Trajectory Prediction
PTP STGCN
SGCN



Nahush’s test time changes that improved performance:
Reflip trajectory and take results
Clamping the court
Agent-specific

Training time experiments:
Data augmentation: flipping x,y

Model evaluation metric:
min_ADE while choosing model parameters


—


Report Material and what we've done:

Baselines (GRU, Transformer)
Graph approaches
Fixed graph
STCGNN, modified STCGNN
Ablations - hyperparam tuning, augmentations, ball coordinates
Hypergraph
Heterogenous Hypergraph  (key inference : hetero agents, CFI)
Groupnet 
MART
Dynamic graph
d-NRI
Equivariant networks
Eqmotion 
Ablations/study areas
(angana) Groupnet ablation: with different k
(angana) Mart ablation: different losses (minADE, meanADE, 
Influence of ball (diff networks/weigh losses)
Losses - MSE, minADE (mart), curriculum (abrupt/smooth)
Network capacity, training time, learning rate schedules (mart, rohan)
Hoop coordinates (and other landmarks)
Inference tricks- TTA, Ensembling, choosing the trajectory (clustering, mean, median, etc)
Hyperparameter tuning
CFI
Learnable per-agent embeddings 

Stgcnn
Base model with nll-loss
Base model with mse loss
Base model + velocity,acceleration + edge weights
TTA, clamping, hyperparam tuning
