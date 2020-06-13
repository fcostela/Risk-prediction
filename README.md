# Risk-prediction
*Rationale*

Many studies have found that eye movement behavior provides a real-time index of the mental activity. Risk management architectures embedded in autonomous vehicles fail to include human cognitive aspects. I set out to evaluate whether eye movements are able to predict risk situations.during a risk detection task while watching driving videos. I created predictive models in Python using logistic regressions and feed-forward neural networks.

I am using a dataset where thirty-two normally sighted subjects (15 female) saw 20 clips of recorded driving scenes while their gaze was tracked. They reported when they considered the car should brake anticipating any hazard. I applied both a mixed-effect logistic regression model and feedforward neural networks between hazard reports and eye movement descriptors. 

All subjects reported at least one major collision hazard in each video (average 3.5 reports). I found that hazard situations were predicted by larger saccades, more and longer fixations, fewer blinks, and a smaller gaze dispersion in both X and Y dimensions. Performance between models incorporating a different combination of descriptors was compared running a test equality of receiver operating characteristic areas. Accuracy using feedforward neural networks outperformed logistic regressions. The model including saccadic magnitude, fixation duration, dispersion in x, and pupil returned the highest ROC area under the curve (0.73). 

Included in the repository:
Python and Matlab code to load eye movement features and predict risk windows on a hazard identification task with driving videos. Compared prediction accuracy and AUC between logistic regressions, feed-forward neural networks, and decision trees.
The model with neural networks outperformed both others. 

Variables included are:

1) A Risk status: positive or negative.

2) Demographic variables: age, race, gender.

3) Eye movement variables: Saccadic magnitude, number of fixations, fixation duration, dispersion in x, dispersion in y, number of blinks, pupil size

