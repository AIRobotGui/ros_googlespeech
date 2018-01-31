#!/usr/bin/env python

from __future__ import division

import argparse
import collections
import itertools
import re
import sys
import threading
import time
import signal

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from google import gax
import grpc
import pyaudio
from six.moves import queue
import six

import rospy
from ros_googlespeech.msg import Utterance


def duration_to_secs(duration):
    return duration.seconds + (duration.nanos / float(1e9))

class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk_size):
        self._rate = rate
        self._chunk_size = chunk_size

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

        # Some useful numbers
        self._num_channels = 1  # API only supports mono for now

    def __enter__(self):
        self.closed = False

        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=self._num_channels, rate=self._rate,
            input=True, frames_per_buffer=self._chunk_size,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        return self

    def __exit__(self, type, value, traceback):
        self.closed = True
        self._audio_stream.close()
        self._buff.put(None)
        self._audio_interface.terminate()


    def _fill_buffer(self, in_data, *args, **kwargs):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue


    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty, KeyboardInterrupt:
                    break

            yield b''.join(data)


class ResumableMicrophoneStream(MicrophoneStream):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk_size, max_replay_secs=5):
        super(ResumableMicrophoneStream, self).__init__(rate, chunk_size)
        self._max_replay_secs = max_replay_secs

        # Some useful numbers
        # 2 bytes in 16 bit samples
        self._bytes_per_sample = 2 * self._num_channels
        self._bytes_per_second = self._rate * self._bytes_per_sample

        self._bytes_per_chunk = (self._chunk_size * self._bytes_per_sample)
        self._chunks_per_second = (
                self._bytes_per_second / self._bytes_per_chunk)
        self._untranscribed = collections.deque(
                maxlen=self._max_replay_secs * self._chunks_per_second)

    def on_transcribe(self, end_time):
        while self._untranscribed and end_time > self._untranscribed[0][1]:
            self._untranscribed.popleft()


    def generator(self, resume=False):
        total_bytes_sent = 0
        if resume:
            # Make a copy, in case on_transcribe is called while yielding them
            catchup = list(self._untranscribed)
            # Yield all the untranscribed chunks first
            for chunk, _ in catchup:
                yield chunk

        for byte_data in super(ResumableMicrophoneStream, self).generator():
            # Populate the replay buffer of untranscribed audio bytes
            total_bytes_sent += len(byte_data)
            chunk_end_time = total_bytes_sent / self._bytes_per_second
            self._untranscribed.append((byte_data, chunk_end_time))

            yield byte_data


def _record_keeper(responses, stream):
    """Calls the stream's on_transcribe callback for each final response.

    Args:
        responses - a generator of responses. The responses must already be
            filtered for ones with results and alternatives.
        stream - a ResumableMicrophoneStream.
    """
    for r in responses:
        result = r.results[0]
        if result.is_final:
            top_alternative = result.alternatives[0]
            # Keep track of what transcripts we've received, so we can resume
            # intelligently when we hit the deadline
            stream.on_transcribe(duration_to_secs(
                    top_alternative.words[-1].end_time))
        yield r




def listen_print_loop_finite(responses):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite         try:it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        top_alternative = result.alternatives[0]
        transcript = top_alternative.transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + '\r')
            sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:
            print(transcript + overwrite_chars)

            num_chars_printed = 0



def listen_print_loop(responses, stream):
    """Iterates through server responses and prints them.

    Same as in transcribe_streaming_mic, but keeps track of when a sent
    audio_chunk has been transcribed.
    """
    with_results = (r for r in responses if (r.results and r.results[0].alternatives))
    listen_print_loop_finite(_record_keeper(with_results, stream))




def listen_publish_loop_finite(responses):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite         try:it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    num_chars_printed = 0

    word_array = []

    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        top_alternative = result.alternatives[0]

        transcript = top_alternative.transcript
        transcript_words = transcript.split(' ')

        new_words = [item for item in transcript_words if item not in word_array]

        word_array.extend(new_words)

        for word in new_words:
            utterance = Utterance()
            utterance.data = word.strip()
            utterance.confidence = result.stability
            word_publisher.publish(utterance)

            print("publishing new word %s %.2f" % (utterance.data, utterance.confidence))

        if result.is_final:
            utterance = Utterance()
            utterance.data = transcript.strip()
            utterance.confidence = top_alternative.confidence
            sentence_publisher.publish(utterance)

            print("publishing sentence %s %.2f" % (utterance.data, utterance.confidence))






def listen_publish_loop(responses, stream):
    """Iterates through server responses and prints them.

    Same as in transcribe_streaming_mic, but keeps track of when a sent
    audio_chunk has been transcribed.
    """
    with_results = (r for r in responses if (r.results and r.results[0].alternatives))
    listen_publish_loop_finite(_record_keeper(with_results, stream))





# Register for Sigint due to generator issues...
def signal_handler(signal, frame):
        print('Exit gracefully...')
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)





# Initialize ROS stuff
rospy.init_node('gspeech_node', anonymous=True)
# Publishers
word_publisher = rospy.Publisher('/stt/words', Utterance, queue_size=10)
sentence_publisher = rospy.Publisher('/stt/sentences', Utterance, queue_size=10)

# See http://g.co/cloud/speech/docs/languages
# for a list of supported languages.
language_code = 'en-US'  # a BCP-47 language tag

client = speech.SpeechClient()
config = types.RecognitionConfig(
    encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code=language_code,
    max_alternatives=1,
    enable_word_time_offsets=True)
streaming_config = types.StreamingRecognitionConfig(
    config=config,
    interim_results=True)


mic_manager = ResumableMicrophoneStream(16000, int(16000 / 10))


with mic_manager as stream:
    resume = False
    while True:

        audio_generator = stream.generator(resume=resume)
        requests = (types.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)
        responses = client.streaming_recognize(streaming_config, requests)

        try:
            # Now, put the transcription responses to use.
            #listen_print_loop(responses, stream)
            listen_publish_loop(responses, stream)

        except grpc.RpcError, e:
            """
            if e.code() not in (grpc.StatusCode.INVALID_ARGUMENT,
                                grpc.StatusCode.OUT_OF_RANGE):
                raise
            details = e.details()
            if e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                if 'deadline too short' not in details:
                    raise
            else:
                if 'maximum allowed stream duration' not in details:
                    raise
            """
            print('Resuming..')
            resume = True
