# AcapellaBot
Isolating vocals from music with a Convolutional Neural Network.  Blog post is [here](http://www.madebyollin.com/posts/cnn_acapella_extraction/).

![](https://d2ppvlu71ri8gs.cloudfront.net/items/2w3t3x1A112i3Y330J2B/output2.gif)

<strong>To Use:</strong>

<ol>
<li><a href="https://github.com/madebyollin/acapellabot/archive/master.zip">Download the repo</a></li>
<li>Install the latest versions of Theano, <a href="https://keras.io/">Keras</a>,<a href="https://github.com/librosa/librosa">librosa</a>, and <a href="http://www.h5py.org/">h5py</a> on Python 3.</li>
<li>Set your dimension ordering to `tf` in `~/keras/keras.json`:
```
{
    "backend": "theano",
    "image_dim_ordering": "tf"
}
```
</li>
<li>Run <code>python acapellabot.py song.mp3</code></li>
</ol>

Enjoy ❤
