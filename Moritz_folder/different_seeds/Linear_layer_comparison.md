# Comparison between uncertainty and weights in linear layer



#### why we use a small precision: better contrast in visualization

<img src="std_vis_s=10p=00001.png" alt="std_vis_s=10p=00001" style="zoom: 67%;" /><img src="std_vis_s=10p=0001.png" alt="std_vis_s=10p=0001" style="zoom:67%;" /><img src="std_vis_s=10p=001.png" alt="std_vis_s=10p=001" style="zoom:67%;" /><img src="std_vis_s=10p=01.png" alt="std_vis_s=10p=01" style="zoom:67%;" /><img src="std_vis_s=10p=1.png" alt="std_vis_s=10p=1" style="zoom:67%;" /><img src="std_vis_s=10p=1000.png" alt="std_vis_s=10p=1000" style="zoom:67%;" /> 

######in high precision (e.g. 1000), every value is identical:

```
tensor([[0.0316, 0.0316, 0.0316,  ..., 0.0316, 0.0316, 0.0316],
        [0.0316, 0.0316, 0.0316,  ..., 0.0316, 0.0316, 0.0316],
        [0.0316, 0.0316, 0.0316,  ..., 0.0316, 0.0316, 0.0316],
        ...,
        [0.0316, 0.0316, 0.0316,  ..., 0.0316, 0.0316, 0.0316],
        [0.0316, 0.0316, 0.0316,  ..., 0.0316, 0.0316, 0.0316],
        [0.0316, 0.0316, 0.0316,  ..., 0.0316, 0.0316, 0.0316]])
```

```
mean standard deviation of layer 0.weight: 0.0316
mean standard deviation of layer 0.bias: 0.0316
mean standard deviation of layer 3.weight: 0.0316
mean standard deviation of layer 3.bias: 0.0316
mean standard deviation of layer 7.weight: 0.0316
mean standard deviation of layer 7.bias: 0.0316
```





#### Following comparisons are using precision 0.0001

------------------



####  seed=10

<img src="std_vis_s=10p=00001.png" alt="std_vis_s=10p=00001" style="zoom: 90%;" /><img src="linear_weights_s=10.png" alt="linear_weights_s=10" style="zoom:90%;" />



<img src="linear_hist_s=10.png" alt="linear_hist_s=10" style="zoom:90%;" />

### histogram of the weights at interesting features (16, 19, 31)

<img src="hist_f16_s10.png" alt="hist_f16_s10" style="zoom:67%;" /><img src="hist_f19_s10.png" alt="hist_f19_s10" style="zoom:67%;" /><img src="hist_f31_s10.png" alt="hist_f31_s10" style="zoom:67%;" />

--------------------------



#### seed=9

<img src="std_vis_s=9p=00001.png" alt="std_vis_s=9p=00001" style="zoom: 90%;" /><img src="linear_weights_s=9.png" alt="linear_weights_s=9" style="zoom:90%;" />



<img src="linear_hist_s=9.png" alt="linear_hist_s=9" style="zoom:90%;" />

### histogram of the weights at interesting features (2, 9, 14, 15, 18, 25)

<img src="hist_f2_s9.png" alt="hist_f2_s9" style="zoom:67%;" /><img src="hist_f9_s9.png" alt="hist_f9_s9" style="zoom:67%;" /><img src="hist_f14_s9.png" alt="hist_f14_s9" style="zoom:67%;" />



<img src="hist_f15_s9.png" alt="hist_f15_s9" style="zoom:67%;" /><img src="hist_f18_s9.png" alt="hist_f18_s9" style="zoom:67%;" /><img src="hist_f25_s9.png" alt="hist_f25_s9" style="zoom:67%;" />

-------------------



#### seed=8

<img src="std_vis_s=8p=00001.png" alt="std_vis_s=8p=00001" style="zoom: 90%;" /><img src="linear_weights_s=8.png" alt="linear_weights_s=8" style="zoom:90%;" />

![linear_hist_s=8](linear_hist_s=8.png)

### histogram of the weights at interesting features (21)

<img src="hist_f21_s8.png" alt="hist_f21_s8" style="zoom:67%;" />

--------------



#### seed = 7

<img src="std_vis_s=7p=00001.png" alt="std_vis_s=7p=00001" style="zoom:90%;" /><img src="linear_weights_s=7.png" alt="linear_weights_s=7" style="zoom:90%;" />

<img src="linear_hist_s=7.png" alt="linear_hist_s=7" style="zoom:90%;" />

### histogram of the weights at interesting features (6, 8, 15, 32)

<img src="hist_f6_s7.png" alt="hist_f6_s7" style="zoom:67%;" /><img src="hist_f8_s7.png" alt="hist_f8_s7" style="zoom:67%;" />![hist_f15_s7](hist_f15_s7.png)<img src="hist_f8_s7.png" alt="hist_f8_s7" style="zoom:67%;" />![hist_f15_s7](hist_f15_s7.png)

<img src="hist_f32_s7.png" alt="hist_f32_s7" style="zoom:67%;" />

-------------



#### seed = 6

<img src="std_vis_s=6p=00001.png" alt="std_vis_s=6p=00001" style="zoom: 90%;" /><img src="linear_weights_s=6.png" alt="linear_weights_s=6" style="zoom:90%;" />

<img src="linear_hist_s=6.png" alt="linear_hist_s=6" style="zoom:90%;" />

### histogram of the weights at interesting features (20, 23, 31)

<img src="hist_f20_s6.png" alt="hist_f20_s6" style="zoom:67%;" /><img src="hist_f23_s6.png" alt="hist_f23_s6" style="zoom:67%;" /><img src="
hist_f31_s6.png" alt="hist_f31_s6" style="zoom:67%;" />

