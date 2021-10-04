# What you'll find

Here is some code that 'works'.  

In another_lin_reg.py you'll find LpRegression - which takes the pth norm and adds it to the cost.  Note: you should only use even powers, unless you know for sure your weights will never be negative.  If your weights go negative, you exit the domain because the pth norm is:

$$ (\sum_{i=0}^{n} x_{i})^{p})^{\frac{1}{p}} $$

For say $x^{\frac{1}{5}}$ we have no domain when x < 0. 

Happy experimenting!

Note: I got the code working, but almost all of the credit goes to: 

https://towardsdatascience.com/ml-from-scratch-linear-polynomial-and-regularized-regression-models-725672336076

I do have an equivalent, but not as well written version.  And it doesn't have the norm stuff.

Anywho, hope this is useful.