from tf_unet import unet, util, image_util

data_provider=image_util.ImageDataProvider("E:/GitHub/tf_unet/img/train/*.tif");
output_path="model"
net=unet.Unet(features_root=64,channels=1,n_class=9);
trainer=unet.Trainer(net, 16);
path=trainer.train(data_provider,output_path,training_iters=32,epochs=100);

prediction=net.predict(path,data);
unet.error_rate(prediction,util.crop_to_shape(label,prediction.shape));
img=util.combine_img_prediction(data,label,prediction);
util.save_image(img,"prediction.jpg");
imread()