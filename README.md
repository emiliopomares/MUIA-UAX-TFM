# MUIAUAX-TFM
TFM Master Universitario en Inteligencia Artificial UAX

Este repositorio contiene todo el código utilizado en el desarrollo del TFM para el MUIA UAX 2023, 2024 de Emilio Pomares


This repository contains the code for the Final Project of the Master's Degree in Artificial Intelligence from Universidad Alfonso X el Sabio.

The goal of the project is to train a CNN-based network to translate a pair of stereo images coming from a stereo camera
into a voxel-based 3D occupancy grid. Applications range from obstacle navigation to robotic manipulation.

This image shows how the working space is divided into a 3D grid of cells which can be occupied or not.
<img src="media/stereo-setup-1.jpg" alt="Description of image" style="width: 600px;">

The goal of the model is to predict as **occupied** only the cells which overlap objects in the working area.
<img src="media/stereo-setup-2.jpg" alt="Description of image" style="width: 300px;">

The network is a U-Net inspired CNN that processes 2D images in the encoding section, and reconstructs 3D images in the decoding. Residual connections allow information to flow from the encoding layers into the decoding ones.
![architecture](media/vxl-net-architecture.jpg)

Check out the **demo video** to see the model working in real-time:

<a href="https://www.youtube.com/watch?v=Fo1N7kAt3Ng">
  <img src="https://img.youtube.com/vi/Fo1N7kAt3Ng/maxresdefault.jpg" alt="Project demo" width="800">
</a>
