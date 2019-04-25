from argparse import ArgumentParser
from youtube_transcript_api import YouTubeTranscriptApi
import fastText
import sys
import re
import time
import datetime
import numpy as np
import math

def pretty_date(seconds):
    return str(datetime.timedelta(seconds=math.ceil(seconds)))

def preprocess(transcript):
    lines = []
    times = [{
        'start': 0.0
    }]
    buf = ''
    count = 0
    duration = 0
    for i in range(len(transcript)):
        obj = transcript[i]
        buf += obj['text'] + ' '
        if count == 3:
            lines.append(re.sub(r"[^a-zA-Z0-9\s]", "", buf).replace("\n", ""))
            times[-1]['duration'] = duration
            duration = 0
            if i + 1 < len(transcript):
                times.append({
                    'start': transcript[i+1]['start']
                })
            else:
                times.append({
                    'start': transcript[i]['start']
                })
            buf = ''
            count = 0
        count += 1
        duration += obj['duration']
    return lines, times

def main(argv):
    parser = ArgumentParser()
    parser.add_argument("-i", "--id", dest="id", help="video id")
    parser.add_argument("-v", "--v", dest="v", help="verbose output", default=False, action='store_true')
    parser.add_argument("-p", "--perf", dest="perf", help="verbose output", default=False, action='store_true')

    args = parser.parse_args()
    transcript = YouTubeTranscriptApi.get_transcript(args.id)
    lines, times = preprocess(transcript)

    start_time = time.time()
    model = fastText.load_model("model.bin")
    load_time = time.time()

    predictions = []
    confidences = []
    for line in lines:
        prediction = model.predict(line)
        predictions.append(prediction[0])
        confidences.append(prediction[1])
    end_time = time.time()
   
    if args.v:
        for i in range(len(predictions)):
            print(predictions[i], confidences[i], " ", lines[i], times[i])

    sponsor_times = []
    for i in range(len(predictions)):
        if predictions[i][0] == "__label__0":
            if i > 0 and (predictions[i - 1][0] == "__label__0") and ((times[i-1]['start'] + times[i-1]['duration']) > times[i]['start']):
                    sponsor_times[-1]['end'] = pretty_date(times[i]['start'] + times[i]['duration'])
            else:
                sponsor_times.append({
                    'start': pretty_date(times[i]['start']),
                    'end': pretty_date(times[i]['start'] + times[i]['duration'])
                })
    print(sponsor_times)

    if args.perf:
        print("Total Time: {0} (s)".format(end_time - start_time))
        print("Model Load Time: {0} (s)".format(load_time - start_time))
        print("Model Run Time: {0} (s)".format(end_time - load_time))
if __name__ == "__main__":
   main(sys.argv[1:])