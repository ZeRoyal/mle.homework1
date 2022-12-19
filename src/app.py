import os
import traceback
from logger import Logger
import sys

from predict import Predictor
from flask import Flask, request, render_template


SHOW_LOG = True

logger = Logger(SHOW_LOG)
log = logger.get_logger(__name__)

server = Flask(__name__)
server.debug = True

log.info("Loading model")

try:
    clf_model = Predictor()
except Exception:
    print('Exception')
    log.error(traceback.format_exc())
    sys.exit(1)


@server.route('/', methods=['GET', 'POST'])
def index():
    """
    The main procedure which works with users requests:
    GET method used for straight request to service
    POST method used for processing user input and return prediction
    
    both methods returns page with input form and result if POST method was used
    """

    # container for variables to output on the page
    page_context = None
    # if used POST method then we get text input from the request data
    # and predict class by classification model
    if request.method == 'POST':
        for key in clf_model.config.keys():  
            print(key)
        page_context = {
            'input_text': input_text,
            'clf_result': '<br/>'.join(list(clf_model.get_review_score(input_text, from_web=True).values()))
        }
        # write user input to log
        log.info(f"classify the review: '{input_text}'")

    return render_template('index.html', context=page_context)

if __name__ == "__main__":
   server.run()