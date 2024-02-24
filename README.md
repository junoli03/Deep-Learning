# Deep-Learning
To create a new conda environment with Python in version of 3.8, you can use the following command:

```conda create -n my_env python=3.8```

This will create a new conda environment called "my_env". To activate this environment, you can use the following command:

```conda activate my_env```

Once the environment is activated, you can use the conda command to install packages into the environment, for example:

```conda install numpy```

You can also use pip to install Python packages within the isolated environment.

To deactivate the environment, you can use the following command:

```conda deactivate```

And to delete the environment, you can use the following command:

```conda env remove -n my_env```

This will remove the "my_env" environment from your system.

# Set up the tensorflow gpu

```pip install tensorflow-gpu==2.6.0```

To downgrade the version of keras, as it may be error with different version with tensorflow-gpu and keras.

```pip install keras==2.6.0```

Install the Protobuf with version 3.20.0

```pip install protobuf==3.20.0```
