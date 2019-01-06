# AFL with DCGAN on CelebA
<img src="../readme/faces.gif" width="1058px"/> <br>

#### Train model
To train the model, first set the dataset path in `host_configurations/paths.py` file. Then run:<br>
`python train.py --app_name appName --training_scheme all ` <br>
or set the arguments in `config.py` file

#### Test model (Run interactive Demo)
To reproduce paper results, download the [pre pretrained(avaialble soon)]() models <br>
`python dc_gan_app.py --app_name appName`

<img src="./gui/demo.png" width="552px"/> <br>
