from bottle import run, get, post, request, delete
import label_image as li
'''
    Instructions
    1. Train the model and get the retrained_graph.pb and the label file to the scripts directory inside model folder
    2. Copy the image and put it into inference_image folder
    3. Call the get_labels function in the label_image script

    Note: Make sure you have installed tensorflow or run it in the anaconda
'''

@get('/')
def indexPage():
    return '''
        <html>
            <head>
                <title>Upload Image</title>
            </head>
            <body>
                <form method="post" action="/upload" enctype="multipart/form-data">
                    <input type="file" accept="image/png, image/jpeg" name="image" />
                    <input type="submit" value="predict" />
                </form>
            </body>
        </html>
    '''

@post('/upload')
def uploadImage():
    image = request.files.image
    filename = "inference_image/image.jpg"
    with open(filename,'wb') as open_file:
        open_file.write(image.file.read())
    open_file.close()
    return get_top_predictions()



#@get('/predictions')
def get_top_predictions():
    lable_index, labels_list, results_list = li.get_lables("image.jpg")
    selected_list = [{"class": labels_list[i], "probability": float(results_list[i]) } for i in lable_index]
    return {"predictions": selected_list}

run(reloader=True,debug=True)
# run(host='0.0.0.0', port=os.environ.get('PORT', '5000'))
