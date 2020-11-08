import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import numpy as np

#load model
path_model = 'path_too_my_model.h5'
predictor = load_model(path_model)

#initial value
img_height = 180
img_width = 180
class_name = ['001_Aldactone_25', '002_Bufenac_Forte', '003_Hydrochlorothiazide_50', '004_Isosorbid_Nitrate', '005_Melobic_15', '006_Metformin_500', '007_Neoflex', '008_Staren_50', '009_Zimmex_20', '010_Zimmex_40', '011_Bromcolex_8', '012_Carisnose_10', '013_Coprofen_400', '014_MiraxM_10', '015_Thyrosit_50', '016_Utmos_30', '017_BromhexineHci_8', '018_Ciprofloxacin_250', '019_Cohexine_8', '020_Coroxin_150', '021_Domperdone_10', '022_Nortussin_15_100', '023_Pharproxin_500', '024_PseudoephedrineHci_60', '025_Allopurinol_100',
              '026_Asmatol_2', '027_Bestatin_Simvastatin_40', '028_Bromphen_4', '029_CarbocalD_1000', '030_HiDiL_600', '031_Muscalm_50', '032_Neuviplex_25', '033_Onsia_8', '034_Paracetamol_500', '035_PMLVita_100', '036_Xanidine_150', '037_Albenz_200', '038_AMCO_300', '039_AtorvastatinSandoz_40', '040_Diclofenac_25', '041_Hydrozide_50', '042_Methopine_245', '043_Musovon_8', '044_Noflox_400', '045_Painol_300', '046_Phemine_100', '047_Piroxicam_20', '048_ProdarilN_250', '049_SPS_330', '050_Tripsyline_10', '051_Tripsyline_25']

#preprocess the image
def preprocess_image(image_bytes):
    
    img = image_bytes.resize((img_height, img_width))
    print("pass")
    img_array = keras.preprocessing.image.img_to_array(img)
    img_preprocess = tf.expand_dims(img_array, 0)  # Create a batch

    return img_preprocess

#prediction
def predict_image(pre_image):
    predictions = predictor.predict(pre_image)
    score = tf.nn.softmax(predictions[0])
    result_predict = class_name[np.argmax(score)]

    return result_predict
