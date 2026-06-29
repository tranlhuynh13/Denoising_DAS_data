# DAS Seismic Denoising

This repository provides methods for **denoising seismic data recorded with Distributed Acoustic Sensing (DAS)**. DAS systems transform fiber‑optic cables into dense arrays of seismic sensors, producing high‑resolution spatio‑temporal measurements. However, DAS recordings often contain substantial noise.  

This repository implements both **signal‑processing** and **deep learning** approaches for improving the quality of DAS earthquake recordings. It also includes scripts for downloading the datasets and notebooks demonstrating the denoising workflows.

---

# Overview

Distributed Acoustic Sensing (DAS) enables seismic monitoring using existing fiber‑optic cables. A single cable can provide thousands of measurement channels, enabling very dense spatial sampling.  

Despite these advantages, DAS data often suffer from noise caused by environmental sources, instrument characteristics, and other disturbances. Effective denoising is therefore an important preprocessing step before further seismic analysis.

This repository provides two approaches:

- **Integrated Denoising Framework (IDF)** – a signal‑processing pipeline combining structure‑oriented filters.
- **SelfMixed** – a deep learning model that learns noise suppression directly from DAS data.

---

# Data

## Downloading DAS Data

The DAS datasets used in the experiments can be downloaded using the script:

```bash
DAS_data/get_all_silixa.sh
```

## P‑Arrival Timing

P‑wave arrival times for each DAS dataset are provided in the **FORGE DAS Earthquake Catalog**:

https://github.com/ariellellouch/FORGE/blob/master/DAS_EQ_Catalog

You can retrieve the P‑arrival time corresponding to each dataset by matching the dataset event time with the entries in the catalog.

Example workflow:

1. Identify the event or dataset name used in this repository.
2. Locate the corresponding event in the catalog file.
3. Extract the listed **P‑arrival time**.
4. Use the arrival time to extract the time window around the first seismic arrival for visualization, training, or evaluation.

---

## Integrated Denoising Framework (IDF)

The **Integrated Denoising Framework (IDF)** is a signal-processing pipeline designed to suppress noise in DAS seismic recordings by combining multiple filtering and slope-based processing stages.

### Core Implementation

- **`IDF.py`**  
  Main implementation of the Integrated Denoising Framework.  
  This file contains the pipeline that integrates the different processing stages.

- **`SOMF_flatten.py`**  
  Implements slope-oriented median filtering (SOMF) in the **flattened domain**.

- **`SOMF_slope.py`**  
  Implements slope-oriented median filtering directly in the **slope domain**.

### Example / Testing

- **`IDF_test.ipynb`**  
  Jupyter notebook demonstrating how to run and evaluate the IDF denoising workflow on DAS data.

Typical workflow:

1. Load DAS seismic data.
2. Apply the IDF pipeline implemented in `IDF.py`.
3. Internally call `SOMF_flatten.py` and `SOMF_slope.py` for slope-oriented filtering.
4. Visualize and evaluate denoising performance in `IDF_test.ipynb`.

---

## SelfMixed Deep Learning Denoising

**SelfMixed** is a deep-learning-based method designed for denoising DAS seismic data using self-supervised or mixed-noise learning strategies.

### Core Implementation

- **`selfmixed_model.py`**  
  Defines the neural network architecture and associated functions for the SelfMixed denoising model.

### Training / Testing

- **`trainSMdas.ipynb`**  
  Jupyter notebook used to:
  - Train the SelfMixed model
  - Run inference on DAS seismic data
  - Evaluate denoising results

---

## Quick Start

### Run the IDF method

Open and run: **`IDF_test.ipynb`**

This notebook demonstrates the full IDF denoising workflow.

### Train or test the SelfMixed model

Open and run: **`trainSMdas.ipynb`**

This notebook handles model training and evaluation.

---

## Notes

- The **IDF approach** is primarily signal-processing based and does not require model training.
- The **SelfMixed approach** requires training and benefits from GPU acceleration.
- Both methods are designed specifically for **Distributed Acoustic Sensing (DAS) seismic data**.

---

## Citation

### BibTeX

```bibtex

@article{xu2024selfmixed,
  title={SelfMixed: Self-supervised mixed noise attenuation for distributed acoustic sensing data},
  author={Xu, Zitai and Wu, Bangyu and Luo, Yisi and Yang, Liuqing and Chen, Yangkang},
  journal={Geophysics},
  volume={89},
  number={5},
  pages={V415--V436},
  year={2024},
  publisher={Society of Exploration Geophysicists}
}

@article{chen2023denoising,
  title={Denoising of distributed acoustic sensing seismic data using an integrated framework},
  author={Chen, Yangkang and Savvaidis, Alexandros and Fomel, Sergey and Chen, Yunfeng and Saad, Omar M and Wang, Hang and Obou{\'e}, Yapo Abol{\'e} Serge Innocent and Yang, Liuqing and Chen, Wei},
  journal={Seismological Society of America},
  volume={94},
  number={1},
  pages={457--472},
  year={2023}
}

@article{wamriew2021deep,
  title={Deep neural networks for detection and location of microseismic events and velocity model inversion from microseismic data acquired by distributed acoustic sensing array},
  author={Wamriew, Daniel and Pevzner, Roman and Maltsev, Evgenii and Pissarenko, Dimitri},
  journal={Sensors},
  volume={21},
  number={19},
  pages={6627},
  year={2021},
  publisher={MDPI}
}

@book{smith2003dsp,
  title={Digital signal processing: a practical guide for engineers and scientists},
  author={Smith, Steven},
  year={2003},
  publisher={Newnes}
}

@article{cetinkaya2019open,
  title={Open intro statistics. OpenIntro},
  author={Cetinkaya-Rundel, Mine and Diez, David and Barr, Christopher},
  journal={Inc. EE. UU. 422p},
  year={2019}
}

@book{proakis2013dsp,
  title={Digital signal processing: Pearson new international edition},
  author={Proakis, John G and Manolakis, Dimitris G},
  year={2013},
  publisher={Pearson Higher Ed}
}

@book{kubat2021ml,
  title={An Introduction to Machine Learning},
  author={Kubat, Miroslav},
  year={2021},
  publisher={Springer}
}

@article{fomel2002pwd,
  title={Applications of plane-wave destruction filters},
  author={Fomel, Sergey},
  journal={Geophysics},
  volume={67},
  number={6},
  pages={1946--1960},
  year={2002},
  publisher={Society of Exploration Geophysicists}
}

@article{wang2022matlab,
  title={A MATLAB code package for 2D/3D local slope estimation and structural filtering},
  author={Wang, Hang and Chen, Yunfeng and Saad, Omar M and Chen, Wei and Obou{\'e}, Yapo Abol{\'e} Serge Innocent and Yang, Liuqing and Fomel, Sergey and Chen, Yangkang},
  journal={Geophysics},
  volume={87},
  number={3},
  pages={F1--F14},
  year={2022},
  publisher={Society of Exploration Geophysicists}
}

@article{beck2009fista,
  title={A fast iterative shrinkage-thresholding algorithm for linear inverse problems},
  author={Beck, Amir and Teboulle, Marc},
  journal={SIAM journal on imaging sciences},
  volume={2},
  number={1},
  pages={183--202},
  year={2009},
  publisher={SIAM}
}

@article{quan2020self2self,
  title={Self2self with dropout: Learning self-supervised denoising from single image},
  author={Quan, Yuhui and Chen, Mingqin and Pang, Tongyao and Ji, Hui},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={1890--1898},
  year={2020}
}

@incollection{imambi2021pytorch,
  title={PyTorch},
  author={Imambi, Sagar and Prakash, Kolla Bhanu and Kanagachidambaresan, GR},
  booktitle={Programming with TensorFlow: solution for edge computing applications},
  pages={87--104},
  year={2021},
  publisher={Springer}
}

@article{python,
  title={Python},
  author={Python, Why},
  journal={Python releases for windows},
  volume={24},
  year={2021},
  publisher={Citeseer}
}

@article{harris2020numpy,
  title={Array programming with NumPy},
  author={Harris, Charles R and Millman, K Jarrod and Van Der Walt, St{\'e}fan J and Gommers, Ralf and Virtanen, Pauli and Cournapeau, David and Wieser, Eric and Taylor, Julian and Berg, Sebastian and Smith, Nathaniel J and others},
  journal={nature},
  volume={585},
  number={7825},
  pages={357--362},
  year={2020},
  publisher={Nature Publishing Group UK London}
}

@article{virtanen2020scipy,
  title={SciPy 1.0: fundamental algorithms for scientific computing in Python},
  author={Virtanen, Pauli and Gommers, Ralf and Oliphant, Travis E and Haberland, Matt and Reddy, Tyler and Cournapeau, David and Burovski, Evgeni and Peterson, Pearu and Weckesser, Warren and Bright, Jonathan and others},
  journal={Nature methods},
  volume={17},
  number={3},
  pages={261--272},
  year={2020},
  publisher={Nature Publishing Group US New York}
}

@article{hunter2007matplotlib,
  title={Matplotlib: A 2D graphics environment},
  author={Hunter, John D},
  journal={Computing in science \& engineering},
  volume={9},
  number={3},
  pages={90--95},
  year={2007},
  publisher={IEEE}
}

@article{fomel2007shaping,
  title={Shaping regularization in geophysical-estimation problems},
  author={Fomel, Sergey},
  journal={Geophysics},
  volume={72},
  number={2},
  pages={R29--R36},
  year={2007},
  publisher={Society of Exploration Geophysicists}
}

@article{hestenes1952cg,
  title={Methods of conjugate gradients for solving linear systems},
  author={Hestenes, Magnus R and Stiefel, Eduard and others},
  journal={Journal of research of the National Bureau of Standards},
  volume={49},
  number={6},
  pages={409--436},
  year={1952}
}

@misc{liu2018pcnn,
      title={Image Inpainting for Irregular Holes Using Partial Convolutions}, 
      author={Guilin Liu and Fitsum A. Reda and Kevin J. Shih and Ting-Chun Wang and Andrew Tao and Bryan Catanzaro},
      year={2018},
      eprint={1804.07723},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1804.07723}, 
}

@misc{he2015delvingdeeprectifierssurpassing,
      title={Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification}, 
      author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
      year={2015},
      eprint={1502.01852},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1502.01852}, 
}

@ARTICLE{wang2018softthresholding,
  author={Wang, Yao and Peng, Jiangjun and Zhao, Qian and Leung, Yee and Zhao, Xi-Le and Meng, Deyu},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Hyperspectral Image Restoration Via Total Variation Regularized Low-Rank Tensor Decomposition}, 
  year={2018},
  volume={11},
  number={4},
  pages={1227-1243},
  keywords={Tensile stress;Image restoration;TV;Gaussian noise;Correlation;Noise measurement;Noise reduction;Hyperspectral image (HSI);low-rank tensor decomposition;mixed noise;total variation (TV)},
  doi={10.1109/JSTARS.2017.2779539}
}

@inproceedings{dabov2009bm3d,
  title={BM3D image denoising with shape-adaptive principal component analysis},
  author={Dabov, Kostadin and Foi, Alessandro and Katkovnik, Vladimir and Egiazarian, Karen},
  booktitle={SPARS'09-Signal Processing with Adaptive Sparse Structured Representations},
  year={2009}
}

@article{chen2023mrr,
  title={Enhancing earthquake detection from distributed acoustic sensing data by coherency measure and moving-rank-reduction filtering},
  author={Chen, Yangkang and Savvaidis, Alexandros and Chen, Yunfeng and Saad, Omar M and Fomel, Sergey},
  journal={Geophysics},
  volume={88},
  number={6},
  pages={WC13--WC23},
  year={2023},
  publisher={Society of Exploration Geophysicists}
}

@article{stein1983bp,
  title={Continuously time-variable recursive digital band-pass filters for seismic signal processing},
  author={Stein, RA and Bartley, NR},
  journal={Geophysics},
  volume={48},
  number={6},
  pages={702--712},
  year={1983},
  publisher={Society of Exploration Geophysicists}
}

@article{yang2023uldnet,
  title={Denoising distributed acoustic sensing data using unsupervised deep learning},
  author={Yang, Liuqing and Fomel, Sergey and Wang, Shoudong and Chen, Xiaohong and Chen, Yangkang},
  journal={Geophysics},
  volume={88},
  number={4},
  pages={V317--V332},
  year={2023},
  publisher={Society of Exploration Geophysicists}
}

@article{yang2023fcdnet,
  title={Denoising of distributed acoustic sensing data using supervised deep learning},
  author={Yang, Liuqing and Fomel, Sergey and Wang, Shoudong and Chen, Xiaohong and Chen, Wei and Saad, Omar M and Chen, Yangkang},
  journal={Geophysics},
  volume={88},
  number={1},
  pages={WA91--WA104},
  year={2023},
  publisher={Society of Exploration Geophysicists}
}

@article{yang2023slknet,
  title={SLKNet: An attention-based deep-learning framework for downhole distributed acoustic sensing data denoising},
  author={Yang, Liuqing and Fomel, Sergey and Wang, Shoudong and Chen, Xiaohong and Chen, Yunfeng and Chen, Yangkang},
  journal={Geophysics},
  volume={88},
  number={6},
  pages={WC69--WC89},
  year={2023},
  publisher={Society of Exploration Geophysicists}
}

@article{Lellouch_2020,
   title={Low‐Magnitude Seismicity With a Downhole Distributed Acoustic Sensing Array—Examples From the FORGE Geothermal Experiment},
   volume={126},
   ISSN={2169-9356},
   url={http://dx.doi.org/10.1029/2020JB020462},
   DOI={10.1029/2020jb020462},
   number={1},
   journal={Journal of Geophysical Research: Solid Earth},
   publisher={American Geophysical Union (AGU)},
   author={Lellouch, A. and Schultz, R. and Lindsey, N.J. and Biondi, B.L. and Ellsworth, W.L.},
   year={2020},
   month=dec 
}

@misc{singh2025gpt5,
      title={OpenAI GPT-5 System Card}, 
      author={Aaditya Singh and Adam Fry and Adam Perelman and Adam Tart and Adi Ganesh and Ahmed El-Kishky and Aidan McLaughlin and Aiden Low and AJ Ostrow and Akhila Ananthram and Akshay Nathan and Alan Luo and Alec Helyar and Aleksander Madry and Aleksandr Efremov and Aleksandra Spyra and Alex Baker-Whitcomb and Alex Beutel and Alex Karpenko and Alex Makelov and Alex Neitz and Alex Wei and Alexandra Barr and Alexandre Kirchmeyer and Alexey Ivanov and Alexi Christakis and Alistair Gillespie and Allison Tam and Ally Bennett and Alvin Wan and Alyssa Huang and Amy McDonald Sandjideh and Amy Yang and Ananya Kumar and Andre Saraiva and Andrea Vallone and Andrei Gheorghe and Andres Garcia Garcia and Andrew Braunstein and Andrew Liu and Andrew Schmidt and Andrey Mereskin and Andrey Mishchenko and Andy Applebaum and Andy Rogerson and Ann Rajan and Annie Wei and Anoop Kotha and Anubha Srivastava and Anushree Agrawal and Arun Vijayvergiya and Ashley Tyra and Ashvin Nair and Avi Nayak and Ben Eggers and Bessie Ji and Beth Hoover and Bill Chen and Blair Chen and Boaz Barak and Borys Minaiev and Botao Hao and Bowen Baker and Brad Lightcap and Brandon McKinzie and Brandon Wang and Brendan Quinn and Brian Fioca and Brian Hsu and Brian Yang and Brian Yu and Brian Zhang and Brittany Brenner and Callie Riggins Zetino and Cameron Raymond and Camillo Lugaresi and Carolina Paz and Cary Hudson and Cedric Whitney and Chak Li and Charles Chen and Charlotte Cole and Chelsea Voss and Chen Ding and Chen Shen and Chengdu Huang and Chris Colby and Chris Hallacy and Chris Koch and Chris Lu and Christina Kaplan and Christina Kim and CJ Minott-Henriques and Cliff Frey and Cody Yu and Coley Czarnecki and Colin Reid and Colin Wei and Cory Decareaux and Cristina Scheau and Cyril Zhang and Cyrus Forbes and Da Tang and Dakota Goldberg and Dan Roberts and Dana Palmie and Daniel Kappler and Daniel Levine and Daniel Wright and Dave Leo and David Lin and David Robinson and Declan Grabb and Derek Chen and Derek Lim and Derek Salama and Dibya Bhattacharjee and Dimitris Tsipras and Dinghua Li and Dingli Yu and DJ Strouse and Drew Williams and Dylan Hunn and Ed Bayes and Edwin Arbus and Ekin Akyurek and Elaine Ya Le and Elana Widmann and Eli Yani and Elizabeth Proehl and Enis Sert and Enoch Cheung and Eri Schwartz and Eric Han and Eric Jiang and Eric Mitchell and Eric Sigler and Eric Wallace and Erik Ritter and Erin Kavanaugh and Evan Mays and Evgenii Nikishin and Fangyuan Li and Felipe Petroski Such and Filipe de Avila Belbute Peres and Filippo Raso and Florent Bekerman and Foivos Tsimpourlas and Fotis Chantzis and Francis Song and Francis Zhang and Gaby Raila and Garrett McGrath and Gary Briggs and Gary Yang and Giambattista Parascandolo and Gildas Chabot and Grace Kim and Grace Zhao and Gregory Valiant and Guillaume Leclerc and Hadi Salman and Hanson Wang and Hao Sheng and Haoming Jiang and Haoyu Wang and Haozhun Jin and Harshit Sikchi and Heather Schmidt and Henry Aspegren and Honglin Chen and Huida Qiu and Hunter Lightman and Ian Covert and Ian Kivlichan and Ian Silber and Ian Sohl and Ibrahim Hammoud and Ignasi Clavera and Ikai Lan and Ilge Akkaya and Ilya Kostrikov and Irina Kofman and Isak Etinger and Ishaan Singal and Jackie Hehir and Jacob Huh and Jacqueline Pan and Jake Wilczynski and Jakub Pachocki and James Lee and James Quinn and Jamie Kiros and Janvi Kalra and Jasmyn Samaroo and Jason Wang and Jason Wolfe and Jay Chen and Jay Wang and Jean Harb and Jeffrey Han and Jeffrey Wang and Jennifer Zhao and Jeremy Chen and Jerene Yang and Jerry Tworek and Jesse Chand and Jessica Landon and Jessica Liang and Ji Lin and Jiancheng Liu and Jianfeng Wang and Jie Tang and Jihan Yin and Joanne Jang and Joel Morris and Joey Flynn and Johannes Ferstad and Johannes Heidecke and John Fishbein and John Hallman and Jonah Grant and Jonathan Chien and Jonathan Gordon and Jongsoo Park and Jordan Liss and Jos Kraaijeveld and Joseph Guay and Joseph Mo and Josh Lawson and Josh McGrath and Joshua Vendrow and Joy Jiao and Julian Lee and Julie Steele and Julie Wang and Junhua Mao and Kai Chen and Kai Hayashi and Kai Xiao and Kamyar Salahi and Kan Wu and Karan Sekhri and Karan Sharma and Karan Singhal and Karen Li and Kenny Nguyen and Keren Gu-Lemberg and Kevin King and Kevin Liu and Kevin Stone and Kevin Yu and Kristen Ying and Kristian Georgiev and Kristie Lim and Kushal Tirumala and Kyle Miller and Lama Ahmad and Larry Lv and Laura Clare and Laurance Fauconnet and Lauren Itow and Lauren Yang and Laurentia Romaniuk and Leah Anise and Lee Byron and Leher Pathak and Leon Maksin and Leyan Lo and Leyton Ho and Li Jing and Liang Wu and Liang Xiong and Lien Mamitsuka and Lin Yang and Lindsay McCallum and Lindsey Held and Liz Bourgeois and Logan Engstrom and Lorenz Kuhn and Louis Feuvrier and Lu Zhang and Lucas Switzer and Lukas Kondraciuk and Lukasz Kaiser and Manas Joglekar and Mandeep Singh and Mandip Shah and Manuka Stratta and Marcus Williams and Mark Chen and Mark Sun and Marselus Cayton and Martin Li and Marvin Zhang and Marwan Aljubeh and Matt Nichols and Matthew Haines and Max Schwarzer and Mayank Gupta and Meghan Shah and Melody Huang and Meng Dong and Mengqing Wang and Mia Glaese and Micah Carroll and Michael Lampe and Michael Malek and Michael Sharman and Michael Zhang and Michele Wang and Michelle Pokrass and Mihai Florian and Mikhail Pavlov and Miles Wang and Ming Chen and Mingxuan Wang and Minnia Feng and Mo Bavarian and Molly Lin and Moose Abdool and Mostafa Rohaninejad and Nacho Soto and Natalie Staudacher and Natan LaFontaine and Nathan Marwell and Nelson Liu and Nick Preston and Nick Turley and Nicklas Ansman and Nicole Blades and Nikil Pancha and Nikita Mikhaylin and Niko Felix and Nikunj Handa and Nishant Rai and Nitish Keskar and Noam Brown and Ofir Nachum and Oleg Boiko and Oleg Murk and Olivia Watkins and Oona Gleeson and Pamela Mishkin and Patryk Lesiewicz and Paul Baltescu and Pavel Belov and Peter Zhokhov and Philip Pronin and Phillip Guo and Phoebe Thacker and Qi Liu and Qiming Yuan and Qinghua Liu and Rachel Dias and Rachel Puckett and Rahul Arora and Ravi Teja Mullapudi and Raz Gaon and Reah Miyara and Rennie Song and Rishabh Aggarwal and RJ Marsan and Robel Yemiru and Robert Xiong and Rohan Kshirsagar and Rohan Nuttall and Roman Tsiupa and Ronen Eldan and Rose Wang and Roshan James and Roy Ziv and Rui Shu and Ruslan Nigmatullin and Saachi Jain and Saam Talaie and Sam Altman and Sam Arnesen and Sam Toizer and Sam Toyer and Samuel Miserendino and Sandhini Agarwal and Sarah Yoo and Savannah Heon and Scott Ethersmith and Sean Grove and Sean Taylor and Sebastien Bubeck and Sever Banesiu and Shaokyi Amdo and Shengjia Zhao and Sherwin Wu and Shibani Santurkar and Shiyu Zhao and Shraman Ray Chaudhuri and Shreyas Krishnaswamy and Shuaiqi and Xia and Shuyang Cheng and Shyamal Anadkat and Simón Posada Fishman and Simon Tobin and Siyuan Fu and Somay Jain and Song Mei and Sonya Egoian and Spencer Kim and Spug Golden and SQ Mah and Steph Lin and Stephen Imm and Steve Sharpe and Steve Yadlowsky and Sulman Choudhry and Sungwon Eum and Suvansh Sanjeev and Tabarak Khan and Tal Stramer and Tao Wang and Tao Xin and Tarun Gogineni and Taya Christianson and Ted Sanders and Tejal Patwardhan and Thomas Degry and Thomas Shadwell and Tianfu Fu and Tianshi Gao and Timur Garipov and Tina Sriskandarajah and Toki Sherbakov and Tomer Kaftan and Tomo Hiratsuka and Tongzhou Wang and Tony Song and Tony Zhao and Troy Peterson and Val Kharitonov and Victoria Chernova and Vineet Kosaraju and Vishal Kuo and Vitchyr Pong and Vivek Verma and Vlad Petrov and Wanning Jiang and Weixing Zhang and Wenda Zhou and Wenlei Xie and Wenting Zhan and Wes McCabe and Will DePue and Will Ellsworth and Wulfie Bain and Wyatt Thompson and Xiangning Chen and Xiangyu Qi and Xin Xiang and Xinwei Shi and Yann Dubois and Yaodong Yu and Yara Khakbaz and Yifan Wu and Yilei Qian and Yin Tat Lee and Yinbo Chen and Yizhen Zhang and Yizhong Xiong and Yonglong Tian and Young Cha and Yu Bai and Yu Yang and Yuan Yuan and Yuanzhi Li and Yufeng Zhang and Yuguang Yang and Yujia Jin and Yun Jiang and Yunyun Wang and Yushi Wang and Yutian Liu and Zach Stubenvoll and Zehao Dou and Zheng Wu and Zhigang Wang},
      year={2025},
      eprint={2601.03267},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.03267}, 
}

@article{riesz1910norm,
  title={Untersuchungen {\"u}ber systeme integrierbarer funktionen},
  author={Riesz, Friedrich},
  journal={Mathematische Annalen},
  volume={69},
  number={4},
  pages={449--497},
  year={1910},
  publisher={Springer}
}

@article{bayes1958essay,
  title={An essay towards solving a problem in the doctrine of chances},
  author={Bayes, Thomas},
  journal={Biometrika},
  volume={45},
  number={3-4},
  pages={296--315},
  year={1958}
}

@article{laplace1774memoire,
  title={M{\'e}moire sur la probabilit{\'e} de causes par les {\'e}venements},
  author={Laplace, Pierre Simon},
  journal={M{\'e}moire de l'acad{\'e}mie royale des sciences},
  year={1774}
}

@inproceedings{nair2010relu,
  title={Rectified linear units improve restricted boltzmann machines},
  author={Nair, Vinod and Hinton, Geoffrey E},
  booktitle={Proceedings of the 27th international conference on machine learning (ICML-10)},
  pages={807--814},
  year={2010}
}

@inproceedings{maas2013rectifier,
  title={Rectifier nonlinearities improve neural network acoustic models},
  author={Maas, Andrew L and Hannun, Awni Y and Ng, Andrew Y and others},
  booktitle={Proc. icml},
  volume={30},
  number={1},
  pages={3},
  year={2013},
  organization={Atlanta, GA}
}

```

