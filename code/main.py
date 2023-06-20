from models import *
from dataproc import *

if __name__ == '__main__':
    
    val_path = 'dataset/validation/'
    train_path = 'dataset/crossmoda2022_training/training_source/'
    
    mask_list, img_list = load_data(train_path,task='segmentation',verbose=1)
    # display_images(img_list, mask_list, num_images=9,random_sampling=True)

 
    model = UNet()
    model.compile()
    model.train(img_list,mask_list,batch_size=32,verbose=1)
    # model.summary()
    model.save("output_models/U-Net.h5")
