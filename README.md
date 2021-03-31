# INTRA-project

The funcs.py file contains all the functions I've used.
The executable.py file can be run from a terminal, it allows different constants and different plots straight from the terminal.

All the functions have some explanation, hopefully it all makes sense. 
The cleaningprocess function will return an array consisting of NaN values. When plotted using pcolormesh, you can see exactly what is deleted on the image. 

The pcolormeshplot function will plot data using pcolormesh.
Set plot to True in badchannelfinder to visualize the band cleaning process and set the limits and cutoff appropriately.
The completeclean function contains the entire cleaning process in one function, but the variables will have to be adjusted to suit.
