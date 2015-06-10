BRISK CVD: Source Code Release v0.0
===================================

License: BSD (see license file included in this folder)

This implementation of BRISK is a adaptation of the original work from 
Stefan Leutenegger, Simon Lynen and Margarita Chli available at: 
https://github.com/rghunter/BRISK using libcvd as an image format. 

It provides with a dynamic library and a sample code showing how to use it. 


This software is an implementation of [1]:  
  [1] Stefan Leutenegger, Margarita Chli and Roland Siegwart, BRISK: 
      Binary Robust Invariant Scalable Keypoints, in Proceedings of the
      IEEE International Conference on Computer Vision (ICCV2011).


Installing the Library
----------------------

Execute the following commands:

```
mkdir build && cd build
cmake ..
make && sudo make install
```
This will build the library and install it. 

Running the BRISK demo 
----------------------