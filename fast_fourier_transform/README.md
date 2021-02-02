## The Fourier Transform   
> A view of the ocean

Let us assume that you're on a beach. As you look into the waves, you notice something. 

You notice that some parts of the ocean are not exactly "wave-like". They don't have those familiar crests and troughs `^\_/^\_/^\_/^\_/^`

This small patch of water has very odd looking wavelets, which somehow look random. 

That random patch is actually a certain mixture of the familiar waves that we all see. So you think:

 ```python
 good_wave + another_good_wave = weird_wave
 ```

If Joseph Fourier was standing beside you, he'd wonder which good waves were used to make up this weird little wave. That is exactly what he did.

He said that given a weird wave, he can express it as a mixture of the natural looking waves that we're all familiar with. 

Fourier said that given a function`f(x)`, we can express it in terms of cosines and sines of increasingly high frequencies. 

<img src = "images/fourier.png" width = "70%">

In other words, fourier series is a way to write `f(x)` in an orthogonal basis of sines and cosines. 

Here's a diagram I made for a better understanding:

<img src = "images/summary_fourier.jpg" width = "100%">