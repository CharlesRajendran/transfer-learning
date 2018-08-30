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
def get_top_predictions():
    lable_index, labels_list, results_list = li.get_lables("bottle.jpg")
    selected_list = [{"class": labels_list[i], "probability": float(results_list[i]) } for i in lable_index]
    return {"predictions": selected_list}

#run(reloader=True,debug=True)
run()
