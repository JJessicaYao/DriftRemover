# Only for the reviewers to check the code detail

This repository is the official implementation of the following paper:

**DriftRemover: Hybrid Energy Optimizations for Anomaly Images Synthesis and Segmentation**<br>

<!-- ![](./main.png)
> **Abstract**<br>
> <font size=3> *This paper tackles the challenge of anomaly image synthesis and segmentation to generate various anomaly images and their segmentation labels to mitigate the issue of data scarcity. Existing approaches employ the precise mask to guide the generation, relying on additional mask generators, leading to increased computational costs and limited anomaly diversity. Although a few works use coarse masks as the guidance to expand diversity, they lack effective generation of labels for synthetic images, thereby reducing their practicality. Therefore, our proposed method simultaneously generates anomaly images and their corresponding masks by utilizing coarse masks and anomaly categories. The framework utilizes attention maps from synthesis process as mask labels and employs two optimization modules to tackle drift challenges, which are mismatches between synthetic results and real situations. Our thorough evaluation showcases the superior performance of our method, improving the pixel-level AP by 1.3\% and  F1-MAX by 1.8\% in anomaly detection tasks on the MVTec dataset. Moreover, the successful application in practical scenarios underscores the framework's practicality and competitive edge, improving the IoU by 37.2\% and F-measure by 25.1\% with the Floor Dirt dataset. The code will be public.* </font>
-->

## Dataset

Download the [MVTec Anomaly Detection (MVTec AD)](https://www.mvtec.com/company/research/datasets/mvtec-ad/) dataset and unzip the archive files under ```./dataset```. (If you wish to try your own datasets, organize the defect-free images, defect images and the corresponding masks in a similar way.)
