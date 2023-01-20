from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import cv2
import os
from werkzeug.utils import secure_filename
app = Flask(__name__)

import tensorflow as tf
import uuid
#model = tf.keras.models.load_model("Model_InceptionV3")
model = tf.keras.models.load_model("Model_CNN")



import json
with open('classes.json', 'r') as f:
    diseases = json.load(f)
    classes = list(diseases.values())
UPLOAD_FOLDER = "static/uploads"

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


import numpy as np
import os


def load_model(image):
    probabilities = model.predict(np.asarray([image]))[0]
    class_idx = np.argmax(probabilities)
    
    return {classes[class_idx]: probabilities[class_idx]}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get("/")
def home():
    return render_template("index.html")

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
@app.post("/")
def predict():
    if 'imagefile' not in request.files:
        flash('No file part')
    imagefile = request.files["imagefile"]
    
    if imagefile.filename == '':
        return render_template('index.html')
       
    if imagefile and allowed_file(imagefile.filename):
        filename = secure_filename(imagefile.filename)
        imagepath = os.path.join("./static/uploads", filename)
        imagefile.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = cv2.imread(imagepath)
        
        img = cv2.resize(img, (299,299))
        #img = cv2.resize(img, (224,224))
        img = img / 255
        classification = load_model(img)
        # PREDICTION
        #prediction ="Disease: %s| Confidence: %.2f" % (list(classification.keys())[0], (list(classification.values())[0])*100) + "%|" 
        state = "%s" %list(classification.keys())[0]
        accuracy ="Confidence: %.2f" % ((list(classification.values())[0])*100) + "%"
        
        
        #flash(prediction + recommendation)
        if (state=='Bacterial leaf blight'):
            return jsonify(rice_state = 'Bacterial leaf blight ' , recommendation=  "RECOMMENDATION/TREATMENT:|Use a variety of plant nutrients in a balanced ratio, particularly nitrogen." + "|Make sure that fields (in conventionally flooded crops) and nurseries maintain good drainage." + "|Keep the fields tidy. Remove weed hosts and plow under volunteer seedlings, straw, rice ratoons, and rice stubble, all of which can act." + "|To reduce disease agents in the soil and plant residues, let empty areas dry out.", confidence=accuracy)
        
        elif (state=='Brown spot'):
            return jsonify(rice_state = 'Brown spot ', recommendation=  "RECOMMENDATION/TREATMENT:|Improving soil fertility is the first step in managing brown spot you need to monitor soil nutrients regularly, apply required fertilizers, for soils that are low in silicon, apply calcium silicate slag before planting" + "|Use fungicides (e.g., iprodione, propiconazole, azoxystrobin, trifloxystrobin, and carbendazim) as seed treatments." + "|Treat seeds with hot water (53−54°C) for 10−12 minutes before planting, to control primary infection at the seedling stage. To increase effectiveness of treatment, pre-soak seeds in cold water for eight hours.", confidence=accuracy)
        
        elif (state=='Healthy'):
            return jsonify(rice_state = 'Healthy ' , recommendation= "Rice leaf is predicted as healthy", confidence=accuracy)
        
        elif (state=='Leaf blast'):
            return jsonify(rice_state = 'Leaf blast ' , recommendation= "RECOMMENDATION/TREATMENT:|Apply nitrogen fertilizer in two or more treatments at once. Fertilizer use in excess might intensify blasts." + "|Flood the field as often as possible." + "|Blasting can be decreased by adding silicon fertilizers to soils that lack silicon, such as calcium silicate. However, silicon needs to be used effectively due to its expensive price. Straws from rice genotypes with high silicon concentration are inexpensive sources of silicon that can be used instead." + "|To effectively suppress blast, systemic fungicides such triazoles and strobilurins might be utilized. Applying a fungicide at heading can help to effectively control the illness.", confidence=accuracy,)
        
        elif (state=='Leaf scald'):
            return jsonify(rice_state = 'Leaf scald ' , recommendation= "RECOMMENDATION/TREATMENT:|Use resistant varieties, Contact your local agriculture office for an up-to-date list of available varieties." + "|Use benomyl, carbendazim, quitozene, and thiophanate-methyl to treat seeds." + "|In the field, spraying of benomyl, fentin acetate, edifenphos, and validamycin significantly reduce the incidence of leaf scald. Foliar application of captafol, mancozeb, and copper oxychloride also reduces the incidence and severity of the fungal disease.", confidence=accuracy)
        
        elif (state=='Narrow brown spot'):
            return jsonify(rice_state = 'Narrow brown spot ' , recommendation= "RECOMMENDATION/TREATMENT:|Keep the fields tidy." + "|To prevent the fungus from finding new hosts and infecting fresh rice crops, eradicate weeds and weedy rice from the field and the surrounding areas." + "|Make sure to get a balanced diet and enough potassium." + "|Spray propiconazole at the booting to heading stages if narrow brown spot poses a risk to the field.",confidence=accuracy)
       
        #return render_template("index.html", filename=filename)
        
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)