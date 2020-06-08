# What if different seeds are used?

* therefore we use the same network and training procedure
* we use the seeds 42, 1107 and 4



### first we will compare the mean variances over the different seeds

###### we compare the means while using a precision of 10, and the means are calculated with the function:

```python
def meancalc(Hessian_diag_x): 
    for ith_tensor in range(len(Hessian_diag_x)) :
        mean = torch.mean(Hessian_diag_x[ith_tensor])
        print("mean variance of layer {0:d}: {1:.4f}".format(ith_tensor+1, mean.item()))
```

* seed = 42:

  * ```python
    precision 10:
    mean variance of layer 1: 0.1018
    mean variance of layer 2: 0.1090
    mean variance of layer 3: 0.1002
    mean variance of layer 4: 0.1009
    mean variance of layer 5: 0.1085
    mean variance of layer 6: 0.1024
    ```

* seed = 1107:

  * ```python
    precision 10:
    mean variance of layer 1: 0.1025
    mean variance of layer 2: 0.1109
    mean variance of layer 3: 0.1003
    mean variance of layer 4: 0.1009
    mean variance of layer 5: 0.1124
    mean variance of layer 6: 0.1025
    ```

* seed = 4:

  * ```python
    precision 10:
    mean variance of layer 1: 0.1025
    mean variance of layer 2: 0.1109
    mean variance of layer 3: 0.1003
    mean variance of layer 4: 0.1009
    mean variance of layer 5: 0.1124
    mean variance of layer 6: 0.1025
    ```

&rarr; the means are very similar, or even the same



### now we visualize the first layer for each of the seeds and try to compare them:

######this is done with the following function in python:

```python
def visualize(tensor):
    output = tensor[0][0]
    for i in range(1, len(tensor[0])):
        output = np.concatenate((output, tensor[0][i]))
    output = output.transpose(2, 0, 1).reshape(5, -1)
    heatmap = sns.heatmap(output)
    plt.xticks = (np.arange(0, step=20))
    plt.show()
```

* seed = 42

  <img src="/Users/moreez/Downloads/seed42" alt="seed=42" style="zoom:100%;" />

* seed = 1107

  <img src="/Users/moreez/Downloads/seed1107" alt="seed=1107" style="zoom:100%;" />

* seed = 4

  <img src="/Users/moreez/Downloads/seed4" alt="seed=4" style="zoom:100%;" />

