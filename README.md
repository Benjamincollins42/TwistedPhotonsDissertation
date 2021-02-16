# TwistedPhotons

This was my project for my (Physics) Masters thesis.
It generated distorted images of 'twisted' photons (specifically of a Laguerre-Gauss variety).
Then it deployed neural networks to classify images, ideally according to its angular mode.

In hindsight there are some issues with this code and the way certain things were implimented due to lack of knowing better at the time.

Things which work well:
+Generation of undistorted Laguerre-Gauss beams
+The functions to propagate the light
+The autoregressive phasescreen functions

Things which worked:
+The generation of phasescreens
 These only worked for some unknown reason using a specific power spectrum for Kolmogorov turbulence.
+Neural networks used
 They worked but in hindsight the structures used were primative.

Bad ideas:
+Storing all the images in batches of 50 in .npy files
 This caused headaches down the line when a data pipeline and augmentation had to be written to deal with this method
 (in comparision to using TensorFlow or Pandas's pre-existing ones).
