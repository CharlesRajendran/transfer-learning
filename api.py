from bottle import run, get, post, request, delete
import label_image as li
import zipfile

'''
    Instructions
    1. Train the model and get the retrained_graph.pb and the label file to the scripts directory inside model folder
    2. Copy the image and put it into inference_image folder
    3. Call the get_labels function in the label_image script

    Note: Make sure you have installed tensorflow or run it in the anaconda
'''

html = '''
    <html>
        <head>
            <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css" rel="stylesheet" crossorigin="anonymous">
        </head>
        <body class="card card-body">
            <form method="post" action="{0}" enctype="multipart/form-data">
                <input type="file" name="{1}" />
                <input class="btn btn-primary" type="submit" value="{2}" />
            </form>
        </body>
    </html>
'''

# pages

@get('/')
def indexPage():
    return html.format("/upload", "image", "Predict")

@get('/upload-model')
def uploadModelPage():
    return html.format("/upload-model", "model", "Upload Model")

# other routes

@post('/upload')
def uploadImage():
    image = request.files.image
    filename = "inference_image/image.jpg"
    _openAndSaveFile(filename, image)

    return _getTopPredictions()

@get('/predictions')
def get_prediction():
    return _getTopPredictions()


@post('/upload-model')
def uploadsModel():
    modelZip = request.files.model
    filename = "model/model.zip"
    _openAndSaveFile(filename, modelZip)

    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall("model/")
    zip_ref.close()

# Utility functions

def _getTopPredictions():
    lable_index, labels_list, results_list = li.get_lables("image.jpg")
    selected_list = [{"class": labels_list[i], "probability": float(results_list[i]) } for i in lable_index]
    return {"predictions": selected_list}

def _openAndSaveFile(filename, uploadedFile):
    with open(filename,'wb') as open_file:
        open_file.write(uploadedFile.file.read())
    open_file.close()


run(reloader=True,debug=True)
# run(host='0.0.0.0', port=os.environ.get('PORT', '5000'))
