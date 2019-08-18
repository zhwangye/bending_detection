# Least Square Method detect line bending or not

the input image is not a simple background, maybe it is complex, for example, bending.jpg. Notice that, the complex is relative, if the image is chaos, the method is not suitable for this condition.(chaos.jpg)



## Steps

1. distance transformation, horizontal projection and vertical projection then get the skeleton image
2. delete one row which the foreground pixel numbers more than 1,in order to exclude the lines in horizontal direction
3. remover the outliers in image
4. least square method fit line and calculate the residual
5. if residual is grater than threshold, output bending info

 