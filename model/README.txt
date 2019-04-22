#################################
### GANerated Hands CNN Model ###
#################################

----------------------------------------------------------------------------------------------------------------------------------------------
|  Terms of use:                                                                                                                             |
|  The software is intended for research purposes only and any use of it for non-scientific and/or commercial means is not allowed.          |
|  This includes publishing any scientific results obtained with our software in non-scientific literature, such as tabloid press.           |
|  If you use this software, you are required to cite the following paper:                                                                   |
|                                                                                                                                            |
|  Mueller F, Bernard F, Sotnychenko O, Mehta D, Sridhar S, Casas D, Theobalt C.                                                             |
|  GANerated Hands for Real-Time 3D Hand Tracking from Monocular RGB.                                                                        |
|  Proc. of IEEE Computer Vision and Pattern Recognition 2018.                                                                               |
|                                                                                                                                            |
|  Refer to the license (license.txt) distributed with the software.                                                                         |
|                                                                                                                                            |
----------------------------------------------------------------------------------------------------------------------------------------------

Important notes:
- You need to add the projection layer (provided in proj_layer_caffe) to your caffe version to be able to run the model.
- The model expects a cropped RGB image of a LEFT hand, scaled in the range from [-1, 1]. If you want to run it on a right hand, you need to flip the image before feeding it to the CNN.
- The order for the 2D heatmaps and the vectorized 3D output is as follows: W, T0, T1, T2, T3, I0, I1, I2, I3, M0, M1, M2, M3, R0, R1, R2, R3, L0, L1, L2, L3. Please also see joints.png for a visual explanation.
- The 3D output is organized as a linear concatenation of the x,y,z-position of every joint (joint1_x, joint1_y, joint1_z, joint2_x, ...) and scaled s.t. the length between wrist and middle finger MCP joint is 1.
