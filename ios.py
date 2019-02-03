import time
import os
import sys

# conversion and inference using coreml won't benefit from GPUs on ubunutu, which might already be in use. 
os.environ["CUDA_VISIBLE_DEVICES"] = ''

import coremltools

def export_to_coreml(model_name) :
    import tfcoreml

    tfcoreml.convert(mlmodel_path = 'frozen/' + model_name + '.mlmodel', 
        tf_model_path = 'frozen/' + model_name + '.pb', 
        output_feature_names = ['output/strided_slice_4:0','output/strided_slice:0'], 
        input_name_shape_dict = {"output/Placeholder:0":[1536,1152,3]},
        image_input_names="output/Placeholder:0")

def convert_to_fp16(modelname) :
    # Load a model, lower its precision, and then save the smaller model.
    file_path = os.path.expanduser('frozen/' + modelname + '.mlmodel')
    model_spec = coremltools.utils.load_spec(file_path)
    model_fp16_spec = coremltools.utils.convert_neural_network_spec_weights_to_fp16(model_spec)
    new_name = file_path.replace(".mlmodel", "_fp16.mlmodel")
    coremltools.utils.save_spec(model_fp16_spec, new_name)

def coreml_predict(model_name) :
    import cv2 as cv
    import numpy as np
    from PIL import Image

    image_file = "test/IMG_1510.jpg"

    core_model = coremltools.models.MLModel('frozen/' + model_name + '.mlmodel')
    cv_image = cv.imread(image_file)
    #resized_image = cv.resize(image, (1152, 1536)) 
    #resized_image = np.asarray(resized_image, dtype = np.float32)
    #resized_image = np.moveaxis(resized_image,-1,0)

    image = Image.open(image_file)
    resized_image = image.resize((1152, 1536), Image.BICUBIC)   
    start = time.time()
    results = core_model.predict({"output__Placeholder__0" : resized_image}) 
    print("Predict in %.3f" % (time.time() - start))

    link_scores = results['output__strided_slice_4__0']
    pixel_scores = results['output__strided_slice__0']
    from inference import display_predictions
    display_predictions(cv_image, link_scores, pixel_scores)


if __name__ == '__main__':
    if len(sys.argv) != 2 :
        print "Usage is python %s <model name>" % sys.argv[0]

    model_name = sys.argv[1]

    export_to_coreml(model_name)
    # convert_to_fp16(model_name)
    #coreml_predict(model_name)
    
